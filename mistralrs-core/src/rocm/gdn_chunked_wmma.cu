// Chunked gated-delta-rule prefill on the gfx12 matrix cores (ROCm only).
//
// The flash-linear-attention decomposition of the recurrence
//   S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - k_t^T S_{t-1} exp(g_t)))^T,  o_t = q_t^T S_t
// over 64-token chunks, in three passes:
//   pass 1 (chunks x heads in parallel): A = tril(beta_i exp(gcum_i - gcum_j) k_i.k_j),
//          T = (I + A)^-1, W = T (beta exp(gcum) K), U = T (beta V), K' = K exp(gtot - gcum)
//   pass 2 (heads x value blocks in parallel, chunks in sequence, the state in registers):
//          Delta_c = U_c - W_c S_c, S_{c+1} = exp(gtot_c) S_c + K'_c^T Delta_c, S_c snapshot out
//   pass 3 (chunks x heads in parallel): O = (exp(gcum) Q) S_c + (tril(Q K^T) * decay) Delta_c
// fp16 operands with f32 accumulation (keys and queries are L2-normalised; the state snapshot
// and the deltas are fp16). Against the f32 recurrence the output is within ~5e-4 relative RMS.
//
// Inputs are the generic recurrence layout: q, k [BH][S][128] f32, v [BH][S][128] f32, g and
// beta [BH][S] f32, output [BH][S][128] f32; the key-major state is [128][128] per state row
// (`slot_indices` map a batch index to its pooled row, negative = padding), in f16/bf16/f32.
// Only head dimensions 128/128 are handled; the launcher returns 0 otherwise.
//
// WMMA 16x16x16 wave32 layouts on gfx12: A lane l holds A[l%16][8(l/16)+i], B lane l holds
// B[8(l/16)+i][l%16], C/D lane l holds D[8(l/16)+j][l%16]. Scratch tiles are stored "packed":
// 32 lanes x 8 values, so a lane fetches its fragment with one 16-byte load. W and K' are packed
// in the A layout, U, Delta and the state snapshot in the D layout (which is the B layout).
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef USE_ROCM

typedef _Float16 cw_h8 __attribute__((ext_vector_type(8)));
typedef float cw_f8 __attribute__((ext_vector_type(8)));
#define CW_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))

namespace {

constexpr int CW_BT = 64;       // tokens per chunk
constexpr int CW_D = 128;       // key and value head dimension
constexpr int CW_LD = 136;      // fp16 row stride of a [64][128] LDS tile (16-byte aligned rows)
constexpr int CW_TLD = 72;      // fp16 row stride of a [64][64] LDS tile
constexpr int CW_THREADS = 256; // 8 waves
constexpr int CW_TILE = 256;    // fp16 values per packed 16x16 tile (32 lanes x 8)
constexpr int CW_TILES_W = 32;  // W, U, K', Delta: 4 token blocks x 8 feature blocks
constexpr int CW_TILES_S = 64;  // state snapshot: 8 key blocks x 8 value blocks

constexpr int CW_STATE_F16 = 0;
constexpr int CW_STATE_BF16 = 1;
constexpr int CW_STATE_F32 = 2;

__device__ __forceinline__ bool cw_padding_row(const int32_t *__restrict__ slot_indices,
                                               int bidx) {
  return slot_indices && slot_indices[bidx] < 0;
}
__device__ __forceinline__ size_t cw_state_row(const int32_t *__restrict__ slot_indices, int bidx,
                                               int h, int num_heads) {
  const size_t slot = slot_indices ? (size_t)slot_indices[bidx] : (size_t)bidx;
  return slot * num_heads + h;
}

__device__ __forceinline__ float cw_state_load(const float *p) { return *p; }
__device__ __forceinline__ float cw_state_load(const __half *p) { return __half2float(*p); }
__device__ __forceinline__ float cw_state_load(const __nv_bfloat16 *p) {
  return __bfloat162float(*p);
}
__device__ __forceinline__ void cw_state_store(float *p, float v) { *p = v; }
__device__ __forceinline__ void cw_state_store(__half *p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void cw_state_store(__nv_bfloat16 *p, float v) {
  *p = __float2bfloat16(v);
}

__device__ __forceinline__ cw_h8 cw_f8_to_h8(const cw_f8 &v) {
  cw_h8 r;
#pragma unroll
  for (int i = 0; i < 8; i++)
    r[i] = (_Float16)v[i];
  return r;
}
__device__ __forceinline__ cw_f8 cw_h8_to_f8(const cw_h8 &v) {
  cw_f8 r;
#pragma unroll
  for (int i = 0; i < 8; i++)
    r[i] = (float)v[i];
  return r;
}
__device__ __forceinline__ cw_f8 cw_zero8() {
  cw_f8 r;
#pragma unroll
  for (int i = 0; i < 8; i++)
    r[i] = 0.0f;
  return r;
}

// Load a 64x128 f32 chunk (rows >= len read as zero) into an fp16 LDS tile, scaling row i by
// row_scale[i] (nullptr: 1).
__device__ __forceinline__ void cw_load_chunk(const float *__restrict__ src, int len,
                                              const float *__restrict__ row_scale,
                                              _Float16 *__restrict__ dst) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int it = 0; it < 8; it++) {
    const int idx = it * CW_THREADS + tid;
    const int row = idx / 32;
    const int c4 = idx % 32;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (row < len)
      v = *reinterpret_cast<const float4 *>(src + (size_t)row * CW_D + c4 * 4);
    const float s = row_scale ? row_scale[row] : 1.0f;
    _Float16 *d = dst + row * CW_LD + c4 * 4;
    d[0] = (_Float16)(v.x * s);
    d[1] = (_Float16)(v.y * s);
    d[2] = (_Float16)(v.z * s);
    d[3] = (_Float16)(v.w * s);
  }
}

// A-operand fragment of tile (row block mi, k block kk) of a row-major fp16 LDS matrix.
__device__ __forceinline__ cw_h8 cw_frag_a(const _Float16 *__restrict__ m, int ld, int mi,
                                           int kk, int lane) {
  return *reinterpret_cast<const cw_h8 *>(m + (mi * 16 + lane % 16) * ld + kk * 16 +
                                          8 * (lane / 16));
}
// B-operand fragment of B = M^T (k block kk, n block ni), M row-major [n][k]: contiguous.
__device__ __forceinline__ cw_h8 cw_frag_bt(const _Float16 *__restrict__ m, int ld, int kk,
                                            int ni, int lane) {
  return *reinterpret_cast<const cw_h8 *>(m + (ni * 16 + lane % 16) * ld + kk * 16 +
                                          8 * (lane / 16));
}
// B-operand fragment of B = M (k block kk, n block ni), M row-major [k][n]: strided.
__device__ __forceinline__ cw_h8 cw_frag_b(const _Float16 *__restrict__ m, int ld, int kk,
                                           int ni, int lane) {
  cw_h8 r;
  const _Float16 *p = m + (kk * 16 + 8 * (lane / 16)) * ld + ni * 16 + lane % 16;
#pragma unroll
  for (int i = 0; i < 8; i++)
    r[i] = p[i * ld];
  return r;
}

// Inclusive prefix sum of g over the chunk (entries past len hold the total).
__device__ __forceinline__ void cw_scan_g(const float *__restrict__ g, int len, float *gcum) {
  if (threadIdx.x == 0) {
    float acc = 0.f;
    for (int t = 0; t < CW_BT; t++) {
      if (t < len)
        acc += g[t];
      gcum[t] = acc;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Pass 1
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(CW_THREADS)
gdn_cw_pass1(const float *__restrict__ k, const float *__restrict__ v,
             const float *__restrict__ g, const float *__restrict__ beta, int seq_len,
             int num_chunks, _Float16 *__restrict__ w_out, _Float16 *__restrict__ u_out,
             _Float16 *__restrict__ kp_out, float *__restrict__ gtot_out) {
  __shared__ __align__(16) _Float16 k16[CW_BT * CW_LD];
  __shared__ __align__(16) _Float16 v16[CW_BT * CW_LD];
  __shared__ __align__(16) float a_s[CW_BT * CW_BT];
  __shared__ __align__(16) _Float16 t16[CW_BT * CW_TLD];
  __shared__ float gcum[CW_BT], bet[CW_BT], bg[CW_BT];

  const int c = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int cs = c * CW_BT;
  const int len = min(CW_BT, seq_len - cs);
  const size_t chunk_id = (size_t)bh * num_chunks + c;

  const float *k_c = k + ((size_t)bh * seq_len + cs) * CW_D;
  const float *v_c = v + ((size_t)bh * seq_len + cs) * CW_D;
  const float *g_c = g + (size_t)bh * seq_len + cs;
  const float *b_c = beta + (size_t)bh * seq_len + cs;

  cw_scan_g(g_c, len, gcum);
  if (tid < CW_BT)
    bet[tid] = tid < len ? b_c[tid] : 0.0f;
  __syncthreads();
  if (tid < CW_BT)
    bg[tid] = bet[tid] * expf(gcum[tid]);
  cw_load_chunk(k_c, len, nullptr, k16);
  cw_load_chunk(v_c, len, bet, v16);
  __syncthreads();
  const float gtot = gcum[CW_BT - 1];
  if (tid == 0)
    gtot_out[chunk_id] = gtot;

  // K' = K exp(gtot - gcum_t), A-packed transposed: tile (kt key block, mi token block), lane l
  // holds K'[mi*16 + 8(l/16) + i][kt*16 + l%16].
  for (int tile = wave; tile < CW_TILES_W; tile += 8) {
    const int kt = tile / 4, mi = tile % 4;
    cw_h8 r;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int t = mi * 16 + 8 * (lane / 16) + i;
      r[i] = (_Float16)((float)k16[t * CW_LD + kt * 16 + lane % 16] * expf(gtot - gcum[t]));
    }
    *reinterpret_cast<cw_h8 *>(kp_out + (chunk_id * CW_TILES_W + tile) * CW_TILE + lane * 8) = r;
  }

  // A = K K^T (16 tiles, two per wave), masked to strictly lower and decayed, in f32.
  for (int tile = wave; tile < 16; tile += 8) {
    const int mi = tile / 4, ni = tile % 4;
    if (ni > mi) {
#pragma unroll
      for (int j = 0; j < 8; j++)
        a_s[(mi * 16 + 8 * (lane / 16) + j) * CW_BT + ni * 16 + lane % 16] = 0.0f;
      continue;
    }
    cw_f8 acc = cw_zero8();
#pragma unroll
    for (int kk = 0; kk < 8; kk++)
      acc = CW_WMMA(cw_frag_a(k16, CW_LD, mi, kk, lane), cw_frag_bt(k16, CW_LD, kk, ni, lane),
                    acc);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int i = mi * 16 + 8 * (lane / 16) + j;
      const int jj = ni * 16 + lane % 16;
      float val = 0.0f;
      if (jj < i && i < len)
        val = bet[i] * acc[j] * expf(gcum[i] - gcum[jj]);
      a_s[i * CW_BT + jj] = val;
    }
  }
  __syncthreads();

  // T = (I + A)^-1 by 16x16 blocks. Diagonal blocks by a 16-step column solve (waves 0..3, one
  // column per lane, x kept in the block's unused upper triangle of a_s); off-diagonal blocks by
  // distance d = I - J: T_IJ = -T_II (sum_{K=J}^{I-1} A_IK T_KJ) on the matrix cores. T lives in
  // t16 (fp16); waves 4..7 zero its strictly upper blocks.
  if (wave < 4) {
    const int I = wave;
    if (lane < 16) {
      const int col = lane;
      float *x = a_s + (I * 16 + col) * CW_BT + I * 16;
      x[col] = 1.0f;
      for (int i = col + 1; i < 16; i++) {
        float sum = 0.0f;
        for (int j = col; j < i; j++)
          sum = fmaf(a_s[(I * 16 + i) * CW_BT + I * 16 + j], x[j], sum);
        x[i] = -sum;
      }
#pragma unroll
      for (int i = 0; i < 16; i++)
        t16[(I * 16 + i) * CW_TLD + I * 16 + col] = (_Float16)(i >= col ? x[i] : 0.0f);
    }
  } else {
    for (int blk = wave - 4; blk < 6; blk += 4) {
      const int I = blk < 3 ? 0 : (blk < 5 ? 1 : 2);
      const int J = blk < 3 ? blk + 1 : (blk < 5 ? blk - 1 : 3);
      for (int e = lane; e < 256; e += 32)
        t16[(I * 16 + e / 16) * CW_TLD + J * 16 + e % 16] = (_Float16)0.0f;
    }
  }
  __syncthreads();
  for (int d = 1; d < 4; d++) {
    if (wave < 4 - d) {
      const int I = wave + d, J = wave;
      cw_f8 x = cw_zero8();
      for (int K = J; K < I; K++) {
        cw_h8 l;
        const float *src = a_s + (I * 16 + lane % 16) * CW_BT + K * 16 + 8 * (lane / 16);
        const float4 lo = *reinterpret_cast<const float4 *>(src);
        const float4 hi = *reinterpret_cast<const float4 *>(src + 4);
        l[0] = (_Float16)lo.x;
        l[1] = (_Float16)lo.y;
        l[2] = (_Float16)lo.z;
        l[3] = (_Float16)lo.w;
        l[4] = (_Float16)hi.x;
        l[5] = (_Float16)hi.y;
        l[6] = (_Float16)hi.z;
        l[7] = (_Float16)hi.w;
        x = CW_WMMA(l, cw_frag_b(t16, CW_TLD, K, J, lane), x);
      }
      const cw_f8 t = CW_WMMA(cw_frag_a(t16, CW_TLD, I, I, lane), cw_f8_to_h8(x), cw_zero8());
#pragma unroll
      for (int j = 0; j < 8; j++)
        t16[(I * 16 + 8 * (lane / 16) + j) * CW_TLD + J * 16 + lane % 16] = (_Float16)(-t[j]);
    }
    __syncthreads();
  }

  // K rows scaled by beta_i exp(gcum_i) for W.
  for (int idx = tid; idx < CW_BT * CW_D; idx += CW_THREADS) {
    const int row = idx / CW_D, col = idx % CW_D;
    k16[row * CW_LD + col] = (_Float16)((float)k16[row * CW_LD + col] * bg[row]);
  }
  __syncthreads();

  // W = T (beta exp(gcum) K) and U = T (beta V): wave w owns feature block w.
  float *stage = a_s + wave * 256; // per-wave 16x16 f32 transpose scratch (a_s is free now)
  for (int mi = 0; mi < 4; mi++) {
    cw_f8 accw = cw_zero8(), accu = cw_zero8();
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      const cw_h8 a = cw_frag_a(t16, CW_TLD, mi, kk, lane);
      accw = CW_WMMA(a, cw_frag_b(k16, CW_LD, kk, wave, lane), accw);
      accu = CW_WMMA(a, cw_frag_b(v16, CW_LD, kk, wave, lane), accu);
    }
    *reinterpret_cast<cw_h8 *>(u_out + (chunk_id * CW_TILES_W + mi * 8 + wave) * CW_TILE +
                               lane * 8) = cw_f8_to_h8(accu);
    // W in the A layout: lane l needs W[mi*16 + l%16][wave*16 + 8(l/16) + i].
#pragma unroll
    for (int j = 0; j < 8; j++)
      stage[(8 * (lane / 16) + j) * 16 + lane % 16] = accw[j];
    __builtin_amdgcn_wave_barrier();
    cw_h8 r;
#pragma unroll
    for (int i = 0; i < 8; i++)
      r[i] = (_Float16)stage[(lane % 16) * 16 + 8 * (lane / 16) + i];
    *reinterpret_cast<cw_h8 *>(w_out + (chunk_id * CW_TILES_W + mi * 8 + wave) * CW_TILE +
                               lane * 8) = r;
    __builtin_amdgcn_wave_barrier();
  }
}

// ---------------------------------------------------------------------------------------------
// Pass 2: one wave per (head, value block); chunks in sequence with the state in registers.
// ---------------------------------------------------------------------------------------------
template <typename StateT>
__global__ void __launch_bounds__(32)
gdn_cw_pass2(const _Float16 *__restrict__ w_in, const _Float16 *__restrict__ u_in,
             const _Float16 *__restrict__ kp_in, const float *__restrict__ gtot,
             StateT *__restrict__ state, const int32_t *__restrict__ slot_indices, int num_heads,
             int num_chunks, _Float16 *__restrict__ s_out, _Float16 *__restrict__ d_out) {
  const int bh = blockIdx.x;
  const int vb = blockIdx.y;
  const int lane = threadIdx.x;
  if (cw_padding_row(slot_indices, bh / num_heads))
    return;
  StateT *st = state + cw_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) *
                           CW_D * CW_D;

  cw_f8 s[8];
#pragma unroll
  for (int kt = 0; kt < 8; kt++)
#pragma unroll
    for (int j = 0; j < 8; j++)
      s[kt][j] = cw_state_load(st + (kt * 16 + 8 * (lane / 16) + j) * CW_D + vb * 16 + lane % 16);

  for (int c = 0; c < num_chunks; c++) {
    const size_t chunk_id = (size_t)bh * num_chunks + c;
    cw_f8 acc[4];
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
      acc[mi] = cw_zero8();
#pragma unroll 2
    for (int kt = 0; kt < 8; kt++) {
      const cw_h8 sh = cw_f8_to_h8(s[kt]);
      *reinterpret_cast<cw_h8 *>(s_out + (chunk_id * CW_TILES_S + kt * 8 + vb) * CW_TILE +
                                 lane * 8) = sh;
#pragma unroll
      for (int mi = 0; mi < 4; mi++) {
        const cw_h8 w = *reinterpret_cast<const cw_h8 *>(
            w_in + (chunk_id * CW_TILES_W + mi * 8 + kt) * CW_TILE + lane * 8);
        acc[mi] = CW_WMMA(w, sh, acc[mi]);
      }
    }
    cw_h8 dlt[4];
#pragma unroll
    for (int mi = 0; mi < 4; mi++) {
      const cw_f8 u = cw_h8_to_f8(*reinterpret_cast<const cw_h8 *>(
          u_in + (chunk_id * CW_TILES_W + mi * 8 + vb) * CW_TILE + lane * 8));
      dlt[mi] = cw_f8_to_h8(u - acc[mi]);
      *reinterpret_cast<cw_h8 *>(d_out + (chunk_id * CW_TILES_W + mi * 8 + vb) * CW_TILE +
                                 lane * 8) = dlt[mi];
    }
    const float decay = expf(gtot[chunk_id]);
#pragma unroll 2
    for (int kt = 0; kt < 8; kt++) {
      cw_f8 a = s[kt] * decay;
#pragma unroll
      for (int mi = 0; mi < 4; mi++) {
        const cw_h8 kp = *reinterpret_cast<const cw_h8 *>(
            kp_in + (chunk_id * CW_TILES_W + kt * 4 + mi) * CW_TILE + lane * 8);
        a = CW_WMMA(kp, dlt[mi], a);
      }
      s[kt] = a;
    }
  }
#pragma unroll
  for (int kt = 0; kt < 8; kt++)
#pragma unroll
    for (int j = 0; j < 8; j++)
      cw_state_store(st + (kt * 16 + 8 * (lane / 16) + j) * CW_D + vb * 16 + lane % 16, s[kt][j]);
}

// ---------------------------------------------------------------------------------------------
// Pass 3
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(CW_THREADS)
gdn_cw_pass3(const float *__restrict__ q, const float *__restrict__ k,
             const float *__restrict__ g, int seq_len, int num_chunks,
             const _Float16 *__restrict__ s_in, const _Float16 *__restrict__ d_in,
             const int32_t *__restrict__ slot_indices, int num_heads, float *__restrict__ out) {
  __shared__ __align__(16) _Float16 q16[CW_BT * CW_LD];
  __shared__ __align__(16) _Float16 k16[CW_BT * CW_LD];
  __shared__ __align__(16) _Float16 p16[CW_BT * CW_TLD];
  __shared__ float gcum[CW_BT], gexp[CW_BT];

  const int c = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int cs = c * CW_BT;
  const int len = min(CW_BT, seq_len - cs);
  const size_t chunk_id = (size_t)bh * num_chunks + c;
  float *out_c = out + ((size_t)bh * seq_len + cs) * CW_D;

  if (cw_padding_row(slot_indices, bh / num_heads)) {
    for (int idx = tid; idx < len * CW_D; idx += CW_THREADS)
      out_c[idx] = 0.0f;
    return;
  }

  const float *q_c = q + ((size_t)bh * seq_len + cs) * CW_D;
  const float *k_c = k + ((size_t)bh * seq_len + cs) * CW_D;
  const float *g_c = g + (size_t)bh * seq_len + cs;

  cw_scan_g(g_c, len, gcum);
  cw_load_chunk(q_c, len, nullptr, q16);
  cw_load_chunk(k_c, len, nullptr, k16);
  __syncthreads();
  if (tid < CW_BT)
    gexp[tid] = expf(gcum[tid]);

  // P = tril(Q K^T) * exp(gcum_i - gcum_j) -> p16.
  for (int tile = wave; tile < 16; tile += 8) {
    const int mi = tile / 4, ni = tile % 4;
    cw_f8 acc = cw_zero8();
    if (ni <= mi) {
#pragma unroll
      for (int kk = 0; kk < 8; kk++)
        acc = CW_WMMA(cw_frag_a(q16, CW_LD, mi, kk, lane), cw_frag_bt(k16, CW_LD, kk, ni, lane),
                      acc);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int i = mi * 16 + 8 * (lane / 16) + j;
      const int jj = ni * 16 + lane % 16;
      float val = 0.0f;
      if (jj <= i && i < len)
        val = acc[j] * expf(gcum[i] - gcum[jj]);
      p16[i * CW_TLD + jj] = (_Float16)val;
    }
  }
  __syncthreads();
  // Q rows scaled by exp(gcum_i) for the inter-chunk term.
  for (int idx = tid; idx < CW_BT * CW_D; idx += CW_THREADS) {
    const int row = idx / CW_D, col = idx % CW_D;
    q16[row * CW_LD + col] = (_Float16)((float)q16[row * CW_LD + col] * gexp[row]);
  }
  __syncthreads();

  for (int mi = 0; mi < 4; mi++) {
    cw_f8 acc = cw_zero8();
#pragma unroll
    for (int kt = 0; kt < 8; kt++) {
      const cw_h8 a = cw_frag_a(q16, CW_LD, mi, kt, lane);
      const cw_h8 sh = *reinterpret_cast<const cw_h8 *>(
          s_in + (chunk_id * CW_TILES_S + kt * 8 + wave) * CW_TILE + lane * 8);
      acc = CW_WMMA(a, sh, acc);
    }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      const cw_h8 a = cw_frag_a(p16, CW_TLD, mi, kk, lane);
      const cw_h8 d = *reinterpret_cast<const cw_h8 *>(
          d_in + (chunk_id * CW_TILES_W + kk * 8 + wave) * CW_TILE + lane * 8);
      acc = CW_WMMA(a, d, acc);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int i = mi * 16 + 8 * (lane / 16) + j;
      if (i < len)
        out_c[(size_t)i * CW_D + wave * 16 + lane % 16] = acc[j];
    }
  }
}

template <typename StateT>
void cw_launch(const float *q, const float *k, const float *v, const float *g, const float *beta,
               StateT *state, float *output, int bh, int seq_len, const int32_t *slot_indices,
               int num_heads, _Float16 *w, _Float16 *u, _Float16 *kp, _Float16 *dlt,
               _Float16 *snap, float *gtot, cudaStream_t stream) {
  const int C = (seq_len + CW_BT - 1) / CW_BT;
  gdn_cw_pass1<<<dim3(C, bh), CW_THREADS, 0, stream>>>(k, v, g, beta, seq_len, C, w, u, kp, gtot);
  gdn_cw_pass2<StateT><<<dim3(bh, 8), 32, 0, stream>>>(w, u, kp, gtot, state, slot_indices,
                                                        num_heads, C, snap, dlt);
  gdn_cw_pass3<<<dim3(C, bh), CW_THREADS, 0, stream>>>(q, k, g, seq_len, C, snap, dlt,
                                                        slot_indices, num_heads, output);
}

} // namespace

// Scratch sizes in bytes for `bh` sequences of `seq_len` tokens: W, U, K' and Delta each take
// `chunked_wmma_gdn_tile_bytes`, the state snapshot `chunked_wmma_gdn_snapshot_bytes`, and
// gtot 4 bytes per chunk.
extern "C" int64_t chunked_wmma_gdn_tile_bytes(int bh, int seq_len) {
  const int64_t C = (seq_len + CW_BT - 1) / CW_BT;
  return (int64_t)bh * C * CW_TILES_W * CW_TILE * 2;
}
extern "C" int64_t chunked_wmma_gdn_snapshot_bytes(int bh, int seq_len) {
  const int64_t C = (seq_len + CW_BT - 1) / CW_BT;
  return (int64_t)bh * C * CW_TILES_S * CW_TILE * 2;
}
extern "C" int64_t chunked_wmma_gdn_num_chunks(int seq_len) { return (seq_len + CW_BT - 1) / CW_BT; }

// Returns 1 when launched, 0 when the shapes are not handled (then nothing was written).
extern "C" int chunked_wmma_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g, const float *beta,
    void *state, float *output, int bh, int seq_len, int k_dim, int v_dim,
    const int32_t *slot_indices, int num_heads, int state_dtype, void *w, void *u, void *kp,
    void *dlt, void *snap, float *gtot, int64_t stream) {
  if (k_dim != CW_D || v_dim != CW_D || bh <= 0 || seq_len <= 0)
    return 0;
  const cudaStream_t custream = (cudaStream_t)stream;
  _Float16 *w16 = (_Float16 *)w, *u16 = (_Float16 *)u, *kp16 = (_Float16 *)kp,
           *d16 = (_Float16 *)dlt, *s16 = (_Float16 *)snap;
  if (state_dtype == CW_STATE_F16) {
    cw_launch(q, k, v, g, beta, (__half *)state, output, bh, seq_len, slot_indices, num_heads, w16,
              u16, kp16, d16, s16, gtot, custream);
  } else if (state_dtype == CW_STATE_BF16) {
    cw_launch(q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, slot_indices,
              num_heads, w16, u16, kp16, d16, s16, gtot, custream);
  } else if (state_dtype == CW_STATE_F32) {
    cw_launch(q, k, v, g, beta, (float *)state, output, bh, seq_len, slot_indices, num_heads, w16,
              u16, kp16, d16, s16, gtot, custream);
  } else {
    return 0;
  }
  return 1;
}

#endif // USE_ROCM
