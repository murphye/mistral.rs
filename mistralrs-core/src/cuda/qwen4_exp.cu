// Qwen4-Exp (Qwen3.8-Flash-Next) kernels: hyper-connections, hashed n-gram PLE
// and QSA sparse attention.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define Q4_PAD_SLOT 0xffffffffu
#define Q4_MAX_PLE_NGRAM 8
#define Q4_MAX_PLE_HEADS 64
#define Q4_MAX_CONV_STATE 16
#define Q4_QSA_MAX_INDEX_HEADS 8
#define Q4_QSA_HEAD_DIM 128
#define Q4_ATTN_HEAD_DIM 256
#define Q4_ATTN_WARPS 4
#define Q4_ATTN_MAX_HEADS_PER_WARP 4
#define Q4_ATTN_ITEMS 4

template <typename T> __device__ __forceinline__ float q4_f(T v) {
  return (float)v;
}
template <typename T> __device__ __forceinline__ T q4_t(float v);
template <> __device__ __forceinline__ float q4_t<float>(float v) { return v; }
template <> __device__ __forceinline__ __half q4_t<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 q4_t<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float q4_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float q4_silu(float x) { return x * q4_sigmoid(x); }

__device__ __forceinline__ float q4_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, o);
  }
  return v;
}

// All threads must call; blockDim.x must be a multiple of 32 and <= 1024.
__device__ __forceinline__ float q4_block_sum(float v, float *smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  v = q4_warp_sum(v);
  __syncthreads();
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();
  v = threadIdx.x < warps ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    v = q4_warp_sum(v);
    if (lane == 0) {
      smem[0] = v;
    }
  }
  __syncthreads();
  return smem[0];
}

template <typename T> struct Q4Vec8 {
  T v[8];
};

template <typename T>
__device__ __forceinline__ void q4_load8(const T *__restrict__ p, float *out) {
  const Q4Vec8<T> vec = *reinterpret_cast<const Q4Vec8<T> *>(p);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = q4_f(vec.v[i]);
  }
}

// An OCP e4m3 (fp8) KV cache element: sign, four exponent bits (bias 7), three mantissa bits;
// exponent 0 is subnormal and 0x7f / 0xff are NaN. Decoded in software so the CUDA and HIP
// builds agree bit for bit.
__device__ __forceinline__ float q4_fp8_f(uint8_t v) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                                            \
    (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx950__) || defined(__gfx1250__))
  // These parts convert OCP e4m3 in hardware (gfx942's instruction is the fnuz format).
  return __builtin_amdgcn_cvt_f32_fp8((unsigned)v, 0);
#else
  const int sign = v >> 7, exp = (v >> 3) & 0xF, man = v & 0x7;
  float f;
  if (exp == 0) {
    f = ldexpf((float)man, -9);  // 2^-6 * man / 8
  } else if (exp == 15 && man == 7) {
    f = __int_as_float(0x7fc00000);
  } else {
    f = ldexpf(1.0f + (float)man * 0.125f, exp - 7);
  }
  return sign ? -f : f;
#endif
}

// A cache element as a float: the activation type as is, fp8 decoded and scaled.
template <typename TC> __device__ __forceinline__ float q4_cache_f(TC v, float scale) {
  return q4_f(v);
}
template <> __device__ __forceinline__ float q4_cache_f<uint8_t>(uint8_t v, float scale) {
  return q4_fp8_f(v) * scale;
}

// Eight consecutive cache elements as floats.
template <typename TC>
__device__ __forceinline__ void q4_load8c(const TC *__restrict__ p, float scale, float *out) {
  const Q4Vec8<TC> vec = *reinterpret_cast<const Q4Vec8<TC> *>(p);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = q4_cache_f<TC>(vec.v[i], scale);
  }
}

// The key cache packs `Q4_CACHE_X<TC>` elements (16 bytes) per [block_size] column group.
template <typename TC> struct Q4CacheX {
  static constexpr int value = 16 / (int)sizeof(TC);
};

// Flattened token -> logical sequence mapping. A null tok_seq is the
// rectangular layout: every sequence has q_len consecutive new tokens (decode
// is q_len 1). tok_seq < 0 marks padding.
struct Q4Tokens {
  const int *tok_seq;
  const int *tok_local;
  const int *seq_start;
  const int *seq_len;
  int n_tokens;
  int n_seqs;
  int q_len;
};

__device__ __forceinline__ int q4_seq_of(const Q4Tokens &l, int t) {
  return l.tok_seq ? l.tok_seq[t] : t / l.q_len;
}
__device__ __forceinline__ int q4_local_of(const Q4Tokens &l, int t) {
  return l.tok_seq ? l.tok_local[t] : t % l.q_len;
}
__device__ __forceinline__ int q4_start_of(const Q4Tokens &l, int seq) {
  return l.seq_start ? l.seq_start[seq] : seq * l.q_len;
}
__device__ __forceinline__ int q4_len_of(const Q4Tokens &l, int seq) {
  return l.seq_len ? l.seq_len[seq] : l.q_len;
}

template <typename T>
__global__ void
q4_hc_norm_kernel(const T *__restrict__ x, const float *__restrict__ weight,
                  T *__restrict__ out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const int stream = row % hc;
  const T *xr = x + row * hidden;
  const float *wr = weight + (size_t)stream * hidden;
  T *orow = out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = q4_f(xr[i]);
    ss = fmaf(v, v, ss);
  }
  ss = q4_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = q4_t<T>(q4_f(xr[i]) * inv * wr[i]);
  }
}

template <typename T>
__global__ void
q4_hc_mix_kernel(const T *__restrict__ xn, const T *__restrict__ gate,
                 T *__restrict__ out, size_t total, int hc, int hidden) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const size_t t = idx / hidden;
  const int i = idx - t * hidden;
  float acc = 0.0f;
  for (int c = 0; c < hc; ++c) {
    const size_t o = (t * hc + c) * hidden + i;
    acc = fmaf(q4_f(xn[o]), q4_sigmoid(q4_f(gate[o])), acc);
  }
  out[idx] = q4_t<T>(acc / (float)hc);
}

// res_out[t, c] = res[t, c] + block_out[t] * 2 * sigmoid(inject[t, c] / hc);
// when next_weight is set, xn_out is the grouped RMSNorm of res_out for the
// next mixer.
template <typename T>
__global__ void
q4_hc_combine_kernel(const T *__restrict__ res, const T *__restrict__ block_out,
                     const T *__restrict__ inject, T *__restrict__ res_out,
                     const float *__restrict__ next_weight,
                     T *__restrict__ xn_out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const size_t t = row / hc;
  const int stream = row - t * hc;
  const float w = 2.0f * q4_sigmoid(q4_f(inject[row]) / (float)hc);
  const T *rr = res + row * hidden;
  const T *br = block_out + t * hidden;
  T *orow = res_out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const T v = q4_t<T>(fmaf(q4_f(br[i]), w, q4_f(rr[i])));
    orow[i] = v;
    const float vf = q4_f(v);
    ss = fmaf(vf, vf, ss);
  }
  if (next_weight == nullptr) {
    return;
  }
  ss = q4_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  const float *wr = next_weight + (size_t)stream * hidden;
  T *xr = xn_out + row * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    xr[i] = q4_t<T>(q4_f(orow[i]) * inv * wr[i]);
  }
}

#ifdef USE_ROCM
// hc_combine over a tensor-parallel partial `block_out`: the ranks exchange their
// partials through peer memory (the receive-buffer layout of candle's
// peer_allreduce.cu) and each combines the full sum, so the all-reduce costs no
// launch of its own. Workgroup 0 pushes the whole `[tokens, hidden]` partial to
// every peer's slot for this rank and publishes the epoch (one system-scope
// fence per launch); every workgroup waits for all peers' epochs before reading
// their slots, and the last workgroup to finish advances the epoch. Every
// workgroup of the grid must be resident at once, so `rows` stays small.
#define Q4_PEER_SPIN_TIMEOUT_TICKS 1000000000ull
// Fence between the pushes into peer memory and the flag stores. Peer memory
// needs system scope.
#ifndef Q4_PEER_PUSH_FENCE
#define Q4_PEER_PUSH_FENCE() __threadfence_system()
#endif

typedef unsigned int q4_v4u __attribute__((ext_vector_type(4)));

template <typename T>
__global__ void __launch_bounds__(512) q4_hc_combine_ar_kernel(
    const T *__restrict__ res, const T *__restrict__ block_out,
    const T *__restrict__ inject, T *__restrict__ res_out,
    const float *__restrict__ next_weight, T *__restrict__ xn_out, int hc,
    int hidden, float eps, const uint64_t *__restrict__ peers, int rank,
    int world, int max_elems, int *__restrict__ epoch,
    unsigned int *__restrict__ finished) {
  __shared__ float smem[32];
  __shared__ int ep_s;
  const int rows = gridDim.x;
  const size_t row = blockIdx.x;
  const size_t t = row / hc;
  const int stream = row - t * hc;
  if (threadIdx.x == 0) {
    ep_s = *epoch + 1;
  }
  __syncthreads();
  const int ep = ep_s;
  const int half = ep & 1;
  if (blockIdx.x == 0) {
    // Push the whole partial, 16 bytes per lane.
    const int nv = (rows / hc) * hidden * (int)sizeof(T) / 16;
    const q4_v4u *src = reinterpret_cast<const q4_v4u *>(block_out);
    for (int peer = 0; peer < world; ++peer) {
      if (peer == rank) {
        continue;
      }
      T *slot = reinterpret_cast<T *>(peers[peer]) +
                (size_t)(half * world + rank) * max_elems;
      q4_v4u *dst = reinterpret_cast<q4_v4u *>(slot);
      for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        dst[i] = src[i];
      }
    }
    Q4_PEER_PUSH_FENCE();
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int peer = 0; peer < world; ++peer) {
        if (peer != rank) {
          int *flags = reinterpret_cast<int *>(peers[world + peer]);
          __hip_atomic_store(flags + rank, ep, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }
  if (threadIdx.x == 0) {
    int *own = reinterpret_cast<int *>(peers[world + rank]);
    const uint64_t start = wall_clock64();
    for (int peer = 0; peer < world; ++peer) {
      if (peer != rank) {
        while (__hip_atomic_load(own + peer, __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < ep) {
          if (wall_clock64() - start > Q4_PEER_SPIN_TIMEOUT_TICKS) {
            __builtin_trap();
          }
        }
      }
    }
  }
  __syncthreads();
  // Sum in rank order so every rank's residual is bit-identical.
  const T *recv = reinterpret_cast<const T *>(peers[rank]) +
                  (size_t)half * world * max_elems + t * hidden;
  const T *br = block_out + t * hidden;
  const float w = 2.0f * q4_sigmoid(q4_f(inject[row]) / (float)hc);
  const T *rr = res + row * hidden;
  T *orow = res_out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float b = 0.0f;
    for (int r = 0; r < world; ++r) {
      b += r == rank ? q4_f(br[i]) : q4_f(recv[(size_t)r * max_elems + i]);
    }
    // The reduced block output is a T tensor on the unfused path; round it the same way.
    const T v = q4_t<T>(fmaf(q4_f(q4_t<T>(b)), w, q4_f(rr[i])));
    orow[i] = v;
    const float vf = q4_f(v);
    ss = fmaf(vf, vf, ss);
  }
  if (next_weight != nullptr) {
    ss = q4_block_sum(ss, smem);
    const float inv = rsqrtf(ss / (float)hidden + eps);
    const float *wr = next_weight + (size_t)stream * hidden;
    T *xr = xn_out + row * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      xr[i] = q4_t<T>(q4_f(orow[i]) * inv * wr[i]);
    }
  }
  // The last workgroup to finish advances the epoch: every workgroup has read it by now.
  if (threadIdx.x == 0) {
    if (atomicAdd(finished, 1u) == (unsigned int)rows - 1) {
      *finished = 0;
      *epoch = ep;
    }
  }
}
#endif

// 36-byte q8_1 block (the MMVQ activation format): d and sum in f16, then 32 int8 values.
struct Q4Q8Block {
  __half d;
  __half s;
  int8_t qs[32];
};

// Indexer query preparation: from the indexer projection `index_qk[t] = [q heads | raw key]`
// (row stride `in_stride`), RMSNorm each query head (weight `norm_w`, Gemma-style `1 + w`
// already applied), rotate its first `2 * half_rot` dims (NeoX halves, cos/sin `[t, half_rot]`)
// and write `index_q[t, h]`; blocks of head 0 also copy the raw key to `raw_key[t]`.
// Grid: (tokens, heads); block: d threads (d <= 256).
template <typename T>
__global__ void q4_index_q_prep_kernel(const T *__restrict__ index_qk, int in_stride,
                                       const T *__restrict__ norm_w, float eps,
                                       const T *__restrict__ cos, const T *__restrict__ sin,
                                       int half_rot, T *__restrict__ index_q,
                                       T *__restrict__ raw_key, int heads, int d) {
  __shared__ float smem[32];
  __shared__ float xs[256];
  const int t = blockIdx.x, h = blockIdx.y, c = threadIdx.x;
  const T *row = index_qk + (size_t)t * in_stride;
  const float v = q4_f(row[h * d + c]);
  const float ss = q4_block_sum(v * v, smem);
  const float inv = rsqrtf(ss / (float)d + eps);
  xs[c] = v * inv * q4_f(norm_w[c]);
  __syncthreads();
  float r = xs[c];
  if (c < half_rot) {
    r = xs[c] * q4_f(cos[(size_t)t * half_rot + c]) - xs[c + half_rot] * q4_f(sin[(size_t)t * half_rot + c]);
  } else if (c < 2 * half_rot) {
    const int j = c - half_rot;
    r = xs[c] * q4_f(cos[(size_t)t * half_rot + j]) + xs[c - half_rot] * q4_f(sin[(size_t)t * half_rot + j]);
  }
  index_q[((size_t)t * heads + h) * d + c] = q4_t<T>(r);
  if (h == 0) {
    raw_key[(size_t)t * d + c] = row[heads * d + c];
  }
}

struct Q4PleHashParams {
  unsigned long long multipliers[Q4_MAX_PLE_NGRAM];
  unsigned long long vocab_sizes[Q4_MAX_PLE_HEADS];
  unsigned long long offsets[Q4_MAX_PLE_HEADS];
  int ngram_size;
  int heads_per_ngram;
  int num_heads;
  unsigned int eos;
};

// hist_pool rows hold the last `hist_len` token ids of each sequence, oldest
// first, stored as id + 1 so a zeroed slot reads as "no token" (EOS).
__global__ void q4_ple_hash_kernel(const unsigned int *__restrict__ tokens,
                                   Q4Tokens l,
                                   const float *__restrict__ hist_pool,
                                   const unsigned int *__restrict__ slots,
                                   int hist_len, Q4PleHashParams p,
                                   long long *__restrict__ rows_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= l.n_tokens) {
    return;
  }
  long long *out = rows_out + (size_t)t * p.num_heads;
  const int seq = q4_seq_of(l, t);
  if (seq < 0) {
    for (int h = 0; h < p.num_heads; ++h) {
      out[h] = (long long)p.offsets[h];
    }
    return;
  }
  const int local = q4_local_of(l, t);
  const unsigned int slot = slots[seq];
  unsigned long long ctx[Q4_MAX_PLE_NGRAM];
  ctx[0] = tokens[t];
  bool cut = false;
  for (int s = 1; s < p.ngram_size; ++s) {
    const int j = local - s;
    bool none = false;
    unsigned int tok = p.eos;
    if (j >= 0) {
      tok = tokens[t - s];
    } else {
      const int h = hist_len + j;
      if (h < 0 || slot == Q4_PAD_SLOT) {
        none = true;
      } else {
        const float stored = hist_pool[(size_t)slot * hist_len + h];
        if (stored <= 0.0f) {
          none = true;
        } else {
          tok = (unsigned int)stored - 1u;
        }
      }
    }
    cut = cut || none || tok == p.eos;
    ctx[s] = cut ? p.eos : tok;
  }
  for (int n = 2; n <= p.ngram_size; ++n) {
    unsigned long long mixed = ctx[0] * p.multipliers[0];
    for (int j = 1; j < n; ++j) {
      mixed ^= ctx[j] * p.multipliers[j];
    }
    const int base = (n - 2) * p.heads_per_ngram;
    for (int g = 0; g < p.heads_per_ngram; ++g) {
      const int h = base + g;
      out[h] = (long long)(mixed % p.vocab_sizes[h] + p.offsets[h]);
    }
  }
}

// New history = last hist_len tokens of (history ++ this chunk), per sequence.
__global__ void
q4_ple_hist_update_kernel(const unsigned int *__restrict__ tokens, Q4Tokens l,
                          float *__restrict__ hist_pool,
                          const unsigned int *__restrict__ slots,
                          int hist_len) {
  const int seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= l.n_seqs) {
    return;
  }
  const unsigned int slot = slots[seq];
  if (slot == Q4_PAD_SLOT) {
    return;
  }
  const int start = q4_start_of(l, seq);
  const int len = q4_len_of(l, seq);
  float *h = hist_pool + (size_t)slot * hist_len;
  float next[Q4_MAX_PLE_NGRAM];
  for (int i = 0; i < hist_len; ++i) {
    const int e = len + i;
    next[i] = e < hist_len ? h[e] : (float)tokens[start + e - hist_len] + 1.0f;
  }
  for (int i = 0; i < hist_len; ++i) {
    h[i] = next[i];
  }
}

// One warp per (token, head): copy a 16-bit row of the n-gram table. The shards
// may be host memory the GPU reads through ATS/HMM.
__global__ void
q4_ple_gather_kernel(const long long *__restrict__ rows, int n_lookups,
                     const unsigned long long *__restrict__ shard_ptrs,
                     const long long *__restrict__ shard_starts, int n_shards,
                     int head_dim, unsigned short *__restrict__ out) {
  const int lookup = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (lookup >= n_lookups) {
    return;
  }
  const long long r = rows[lookup];
  int lo = 0;
  int hi = n_shards - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (shard_starts[mid] <= r) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const unsigned short *src =
      reinterpret_cast<const unsigned short *>(shard_ptrs[lo]) +
      (size_t)(r - shard_starts[lo]) * head_dim;
  unsigned short *dst = out + (size_t)lookup * head_dim;
  for (int i = lane; i < head_dim; i += 32) {
    dst[i] = src[i];
  }
}

// Gather from the device-resident quantized table. Q8/Q4: per row, symmetric
// int8 or offset-8 nibbles with one f16 scale per Q4_PLE_GROUP values in
// `scales`. IQ4_NL: ggml block_iq4_nl rows (f16 scale + 16 nibble bytes per
// Q4_PLE_GROUP values, low nibbles first) straight from a GGUF; `scales` unused.
#define Q4_PLE_GROUP 32
#define Q4_PLE_FMT_Q8 0
#define Q4_PLE_FMT_Q4 1
#define Q4_PLE_FMT_IQ4_NL 2
#define Q4_IQ4_NL_BLOCK_BYTES 18
__constant__ signed char q4_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__global__ void
q4_ple_gather_quant_kernel(const long long *__restrict__ rows, int n_lookups,
                           const unsigned char *__restrict__ data,
                           const __half *__restrict__ scales, int head_dim,
                           int format, __nv_bfloat16 *__restrict__ out) {
  const int lookup = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (lookup >= n_lookups) {
    return;
  }
  const size_t r = (size_t)rows[lookup];
  const int groups = head_dim / Q4_PLE_GROUP;
  __nv_bfloat16 *dst = out + (size_t)lookup * head_dim;
  if (format == Q4_PLE_FMT_IQ4_NL) {
    const unsigned char *src =
        data + r * (size_t)groups * Q4_IQ4_NL_BLOCK_BYTES;
    for (int i = lane; i < head_dim; i += 32) {
      const unsigned char *blk =
          src + (size_t)(i / Q4_PLE_GROUP) * Q4_IQ4_NL_BLOCK_BYTES;
      const int j = i % Q4_PLE_GROUP;
      const unsigned char byte = blk[2 + (j & 15)];
      const int idx = j < 16 ? (byte & 15) : (byte >> 4);
      const float d = __half2float(*reinterpret_cast<const __half *>(blk));
      dst[i] = __float2bfloat16_rn(d * (float)q4_kvalues_iq4nl[idx]);
    }
    return;
  }
  const int bits = format == Q4_PLE_FMT_Q8 ? 8 : 4;
  const size_t row_bytes = (size_t)head_dim * bits / 8;
  const unsigned char *src = data + r * row_bytes;
  const __half *sc = scales + r * groups;
  for (int i = lane; i < head_dim; i += 32) {
    int q;
    if (bits == 8) {
      q = (int)(signed char)src[i];
    } else {
      const unsigned char byte = src[i >> 1];
      q = (int)((i & 1) ? (byte >> 4) : (byte & 15)) - 8;
    }
    dst[i] = __float2bfloat16_rn((float)q * __half2float(sc[i / Q4_PLE_GROUP]));
  }
}

// Per token: gate each hyper-connection stream by the normalized key/query dot
// product, scale the shared value, and RMS-normalize the gated value for the
// conv. One block per token.
template <typename T>
__global__ void
q4_ple_gate_kernel(const T *__restrict__ key,
                   const T *__restrict__ hidden_states,
                   const T *__restrict__ value, const float *__restrict__ w_key,
                   const float *__restrict__ w_query,
                   const float *__restrict__ w_conv, T *__restrict__ gated_out,
                   T *__restrict__ normed_out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  __shared__ float stats[3 * 8 + 1];
  const size_t t = blockIdx.x;
  const T *kr = key + t * hc * hidden;
  const T *hr = hidden_states + t * hc * hidden;
  const T *vr = value + t * hidden;
  float vv = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = q4_f(vr[i]);
    vv = fmaf(v, v, vv);
  }
  vv = q4_block_sum(vv, smem);
  if (threadIdx.x == 0) {
    stats[3 * 8] = vv;
  }
  for (int c = 0; c < hc; ++c) {
    float kk = 0.0f, qq = 0.0f, kq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const size_t o = (size_t)c * hidden + i;
      const float k = q4_f(kr[o]);
      const float q = q4_f(hr[o]);
      kk = fmaf(k, k, kk);
      qq = fmaf(q, q, qq);
      kq = fmaf(k * w_key[o], q * w_query[o], kq);
    }
    kk = q4_block_sum(kk, smem);
    qq = q4_block_sum(qq, smem);
    kq = q4_block_sum(kq, smem);
    if (threadIdx.x == 0) {
      stats[3 * c] = kk;
      stats[3 * c + 1] = qq;
      stats[3 * c + 2] = kq;
    }
  }
  __syncthreads();
  const float mean_vv = stats[3 * 8] / (float)hidden;
  for (int c = 0; c < hc; ++c) {
    const float inv_k = rsqrtf(stats[3 * c] / (float)hidden + eps);
    const float inv_q = rsqrtf(stats[3 * c + 1] / (float)hidden + eps);
    const float g = stats[3 * c + 2] * inv_k * inv_q * rsqrtf((float)hidden);
    const float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
    const float sig = q4_sigmoid(g > 0.0f ? mag : (g < 0.0f ? -mag : 0.0f));
    const float inv_g = rsqrtf(sig * sig * mean_vv + eps);
    T *gr = gated_out + (t * hc + c) * hidden;
    T *nr = normed_out + (t * hc + c) * hidden;
    const float *wc = w_conv + (size_t)c * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float gv = sig * q4_f(vr[i]);
      gr[i] = q4_t<T>(gv);
      nr[i] = q4_t<T>(gv * inv_g * wc[i]);
    }
  }
}

// out = residual + gated + silu(dilated causal depthwise conv(normed)); history
// before the chunk comes from the pooled conv state `[slot, channels,
// state_len]` (oldest first).
template <typename T>
__global__ void
q4_ple_conv_kernel(const T *__restrict__ normed, const T *__restrict__ gated,
                   const T *__restrict__ residual, const T *__restrict__ weight,
                   const T *__restrict__ state_pool,
                   const unsigned int *__restrict__ slots, T *__restrict__ out,
                   Q4Tokens l, int channels, int kernel_size, int dilation,
                   int state_len) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)l.n_tokens * channels) {
    return;
  }
  const int t = idx / channels;
  const int c = idx - (size_t)t * channels;
  const int seq = q4_seq_of(l, t);
  if (seq < 0) {
    out[idx] = residual[idx];
    return;
  }
  const int local = q4_local_of(l, t);
  const unsigned int slot = slots[seq];
  float acc = 0.0f;
  for (int k = 0; k < kernel_size; ++k) {
    const int j = local - (kernel_size - 1 - k) * dilation;
    float x = 0.0f;
    if (j >= 0) {
      x = q4_f(normed[(size_t)(t - (local - j)) * channels + c]);
    } else if (slot != Q4_PAD_SLOT && state_len + j >= 0) {
      x = q4_f(state_pool[((size_t)slot * channels + c) * state_len +
                          state_len + j]);
    }
    acc = fmaf(q4_f(weight[(size_t)c * kernel_size + k]), x, acc);
  }
  out[idx] = q4_t<T>(q4_f(residual[idx]) + q4_f(gated[idx]) + q4_silu(acc));
}

template <typename T>
__global__ void
q4_ple_conv_state_update_kernel(const T *__restrict__ normed,
                                T *__restrict__ state_pool,
                                const unsigned int *__restrict__ slots,
                                Q4Tokens l, int channels, int state_len) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)l.n_seqs * channels) {
    return;
  }
  const int seq = idx / channels;
  const int c = idx - (size_t)seq * channels;
  const unsigned int slot = slots[seq];
  if (slot == Q4_PAD_SLOT) {
    return;
  }
  const int start = q4_start_of(l, seq);
  const int len = q4_len_of(l, seq);
  T *s = state_pool + ((size_t)slot * channels + c) * state_len;
  float next[Q4_MAX_CONV_STATE];
  for (int i = 0; i < state_len; ++i) {
    const int e = len + i;
    next[i] =
        e < state_len
            ? q4_f(s[e])
            : q4_f(normed[(size_t)(start + e - state_len) * channels + c]);
  }
  for (int i = 0; i < state_len; ++i) {
    s[i] = q4_t<T>(next[i]);
  }
}

// QSA aux cache rows are [raw indexer key | cos | sin] per token slot; once a
// block completes, its first token's key becomes the pooled, rotated block key.

struct Q4Paged {
  const int *block_tables;
  const unsigned int *kv_lens;
  int max_blocks_per_seq;
  int block_size;
};

__device__ __forceinline__ long long q4_slot(const Q4Paged &pg, int seq,
                                             int pos) {
  const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq +
                                  pos / pg.block_size];
  return (long long)blk * pg.block_size + pos % pg.block_size;
}

// Absolute kv position of flattened token t, or -1 for padding.
__device__ __forceinline__ int q4_pos_of(const Q4Tokens &l, const Q4Paged &pg,
                                         int t, int &seq) {
  seq = q4_seq_of(l, t);
  if (seq < 0) {
    return -1;
  }
  return (int)pg.kv_lens[seq] - q4_len_of(l, seq) + q4_local_of(l, t);
}

// Writes the token's raw indexer key to its aux slot and, for the first token of each
// compressed block (block sizes are multiples of `ratio`, so slot % ratio == 0), the block's
// cos/sin to `block_cs`: the finalize kernel rotates the pooled key by that position alone.
// Raw keys are stored as int8 with one f32 scale per row (`Q4_QSA_RAW_ROW_BYTES` bytes):
// the pooled, RMS-normed block key tolerates the ~0.4% per-element error. A key is only
// needed until its compressed block completes, so rows live in a per-step scratch (one per
// token of the step) plus a per-sequence carry holding the open block and the one before it
// (speculative rollback reopens at most one block); the carry is indexed by the sequence's
// recurrent state slot as `[(slot * 2 + (pos / ratio) % 2) * ratio + pos % ratio]`.
// 128 threads per token, one element each.
#define Q4_QSA_RAW_ROW_BYTES (Q4_QSA_HEAD_DIM + 4)

// The carry row of position `p` for a sequence whose physical recurrent slot is `slot`:
// physical slots are `logical * lanes + lane`, and the logical slot is what stays with the
// sequence while speculative checkpoints switch lanes.
__device__ __forceinline__ size_t q4_carry_row(unsigned slot, int lanes, int p, int ratio) {
  return ((size_t)(slot / (unsigned)lanes) * 2 + (size_t)((p / ratio) & 1)) * ratio + p % ratio;
}

// The raw-key row of position `p` of sequence `seq`: this step's scratch row when the
// position is one of the step's tokens, the sequence's carry otherwise.
__device__ __forceinline__ const uint8_t *
q4_raw_row(const uint8_t *__restrict__ scratch, const uint8_t *__restrict__ carry,
           const unsigned *__restrict__ state_slots, int lanes, const Q4Tokens &l,
           const Q4Paged &pg, int seq, int p, int aux_dim, int ratio) {
  const int first = (int)pg.kv_lens[seq] - q4_len_of(l, seq);
  if (p >= first) {
    return scratch + (size_t)(q4_start_of(l, seq) + (p - first)) * aux_dim;
  }
  return carry + q4_carry_row(state_slots[seq], lanes, p, ratio) * aux_dim;
}

template <typename T>
__global__ void q4_qsa_aux_write_kernel(
    const T *__restrict__ raw_key, const T *__restrict__ cos,
    const T *__restrict__ sin, const long long *__restrict__ slot_mapping,
    uint8_t *__restrict__ aux, T *__restrict__ block_cs, int n_tokens, int key_dim,
    int half_rot, int aux_dim, int ratio) {
  __shared__ float warp_max[4];
  const int t = blockIdx.x;
  if (t >= n_tokens) {
    return;
  }
  const long long slot = slot_mapping[t];
  if (slot < 0) {
    return;
  }
  // Scratch rows are per step token; only the block cos/sin lives at the KV slot's block.
  uint8_t *row = aux + (size_t)t * aux_dim;
  const int i = threadIdx.x;
  const float x = i < key_dim ? q4_f(raw_key[(size_t)t * key_dim + i]) : 0.0f;
  float amax = fabsf(x);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  }
  if ((i & 31) == 0) {
    warp_max[i >> 5] = amax;
  }
  __syncthreads();
  amax = fmaxf(fmaxf(warp_max[0], warp_max[1]), fmaxf(warp_max[2], warp_max[3]));
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  if (i < key_dim) {
    const float q = rintf(x / scale);
    ((int8_t *)row)[i] = (int8_t)fminf(fmaxf(q, -127.0f), 127.0f);
  }
  if (i == 0) {
    *(float *)(row + key_dim) = scale;
  }
  if (slot % ratio == 0) {
    T *cs = block_cs + (slot / ratio) * (2 * half_rot);
    for (int i = threadIdx.x; i < half_rot; i += blockDim.x) {
      cs[i] = cos[(size_t)t * half_rot + i];
      cs[half_rot + i] = sin[(size_t)t * half_rot + i];
    }
  }
}

// One warp per token; the token that completes a block writes the block key.
// key_dim must be 128.
template <typename T>
__global__ void
q4_qsa_finalize_kernel(const uint8_t *__restrict__ aux, const uint8_t *__restrict__ carry,
                       const unsigned *__restrict__ state_slots, int lanes,
                       const T *__restrict__ block_cs, T *__restrict__ block_keys,
                       Q4Tokens l, Q4Paged pg, const float *__restrict__ norm_weight,
                       int ratio, int half_rot, int aux_dim, float eps) {
  const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (t >= l.n_tokens) {
    return;
  }
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  if (pos < 0 || (pos + 1) % ratio != 0) {
    return;
  }
  const int start = pos + 1 - ratio;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int m = 0; m < ratio; ++m) {
    const uint8_t *row =
        q4_raw_row(aux, carry, state_slots, lanes, l, pg, seq, start + m, aux_dim, ratio);
    const float row_scale = *(const float *)(row + Q4_QSA_HEAD_DIM);
    const int8_t *src = (const int8_t *)row;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] += (float)src[lane * 4 + i] * row_scale;
    }
  }
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // HF pools in f32 and rounds to the checkpoint dtype before the norm
    v[i] = q4_f(q4_t<T>(v[i] / (float)ratio));
    ss = fmaf(v[i], v[i], ss);
  }
  ss = q4_warp_sum(ss);
  const float inv = rsqrtf(ss / (float)Q4_QSA_HEAD_DIM + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = q4_f(q4_t<T>(v[i] * inv * norm_weight[lane * 4 + i]));
  }
  const T *cs = block_cs + q4_slot(pg, seq, start) / ratio * (2 * half_rot);
  // Keep raw keys intact when speculative rollback reopens a completed block.
  T *dst = block_keys + q4_slot(pg, seq, start) / ratio * Q4_QSA_HEAD_DIM;
  const T *sn = cs + half_rot;
  float out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = v[i];
  }
  // rotate_half partner is half_rot dims away
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 4 + i;
    const float partner = __shfl_sync(
        0xffffffff, v[i],
        d < half_rot ? lane + half_rot / 4 : (lane - half_rot / 4) & 31);
    if (d < half_rot) {
      out[i] = v[i] * q4_f(cs[d]) - partner * q4_f(sn[d]);
    } else if (d < 2 * half_rot) {
      out[i] = v[i] * q4_f(cs[d - half_rot]) + partner * q4_f(sn[d - half_rot]);
    }
  }
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    dst[lane * 4 + i] = q4_t<T>(out[i]);
  }
}

// After a step, each sequence's open block (positions from the last block boundary to its
// kv length) is copied from the step's scratch into the carry; positions before this step
// are already there. One block per sequence, one thread per row byte.
__global__ void q4_qsa_carry_update_kernel(const uint8_t *__restrict__ scratch,
                                           uint8_t *__restrict__ carry,
                                           const unsigned *__restrict__ state_slots, int lanes,
                                           Q4Tokens l, Q4Paged pg, int ratio, int aux_dim) {
  const int seq = blockIdx.x;
  if (seq >= l.n_seqs) {
    return;
  }
  const int len = q4_len_of(l, seq);
  if (len <= 0) {
    return;
  }
  const int kv = (int)pg.kv_lens[seq];
  const int first = kv - len;
  const int open_start = kv - kv % ratio;
  for (int p = max(open_start, first); p < kv; ++p) {
    const uint8_t *src = scratch + (size_t)(q4_start_of(l, seq) + (p - first)) * aux_dim;
    uint8_t *dst = carry + q4_carry_row(state_slots[seq], lanes, p, ratio) * aux_dim;
    for (int b = threadIdx.x; b < aux_dim; b += blockDim.x) {
      dst[b] = src[b];
    }
  }
}

// scores[t, j] = sum_h relu(q[t, h] . block_key[j]) / sqrt(d) for rows with
// more complete blocks than the budget selects; other rows are skipped.
template <typename T>
__global__ void
q4_qsa_score_kernel(const T *__restrict__ q, const T *__restrict__ block_keys,
                    Q4Tokens l, Q4Paged pg, float *__restrict__ scores,
                    int score_stride, int n_heads, int ratio, int topk,
                    int key_dim) {
  __shared__ float qs[Q4_QSA_MAX_INDEX_HEADS * Q4_QSA_HEAD_DIM];
  const int t = blockIdx.y;
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  const int nb = pos < 0 ? 0 : (pos + 1) / ratio;
  if (nb <= topk || (int)(blockIdx.x * blockDim.x) >= nb) {
    return;
  }
  for (int i = threadIdx.x; i < n_heads * Q4_QSA_HEAD_DIM; i += blockDim.x) {
    qs[i] = q4_f(q[(size_t)t * n_heads * Q4_QSA_HEAD_DIM + i]);
  }
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nb) {
    return;
  }
  const T *key = block_keys + q4_slot(pg, seq, j * ratio) / ratio * key_dim;
  float dots[Q4_QSA_MAX_INDEX_HEADS];
#pragma unroll
  for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
    dots[h] = 0.0f;
  }
  for (int d = 0; d < Q4_QSA_HEAD_DIM; d += 8) {
    float k[8];
    q4_load8(key + d, k);
#pragma unroll
    for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
      if (h < n_heads) {
        const float *qh = qs + h * Q4_QSA_HEAD_DIM + d;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          dots[h] = fmaf(qh[i], k[i], dots[h]);
        }
      }
    }
  }
  float total = 0.0f;
#pragma unroll
  for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
    if (h < n_heads) {
      total += fmaxf(dots[h], 0.0f);
    }
  }
  scores[(size_t)t * score_stride + j] = total * rsqrtf((float)Q4_QSA_HEAD_DIM);
}

__device__ __forceinline__ unsigned int q4_ordered(float v) {
  if (v != v) {
    return 0u;
  }
  const unsigned int bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// Per query row: the indices of the topk largest scores among the first nb
// complete blocks (all of them when nb <= topk). Radix select over the
// order-preserving float bits.
__global__ void q4_qsa_topk_kernel(const float *__restrict__ scores,
                                   int score_stride, Q4Tokens l, Q4Paged pg,
                                   int ratio, int topk,
                                   int *__restrict__ selected,
                                   int *__restrict__ n_selected) {
  __shared__ unsigned int hist[256];
  __shared__ unsigned int s_prefix, s_mask, s_remaining;
  __shared__ unsigned int s_gt, s_eq;
  const int t = blockIdx.x;
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  const int nb = pos < 0 ? 0 : (pos + 1) / ratio;
  int *out = selected + (size_t)t * topk;
  if (nb <= topk) {
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
      out[i] = i;
    }
    if (threadIdx.x == 0) {
      n_selected[t] = nb;
    }
    return;
  }
  const float *row = scores + (size_t)t * score_stride;
  if (threadIdx.x == 0) {
    s_prefix = 0u;
    s_mask = 0u;
    s_remaining = topk;
  }
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      hist[i] = 0u;
    }
    __syncthreads();
    const unsigned int prefix = s_prefix;
    const unsigned int mask = s_mask;
    for (int j = threadIdx.x; j < nb; j += blockDim.x) {
      const unsigned int key = q4_ordered(row[j]);
      if ((key & mask) == prefix) {
        atomicAdd(&hist[(key >> shift) & 255u], 1u);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int remaining = s_remaining;
      unsigned int above = 0u;
      int digit = 255;
      for (; digit > 0; --digit) {
        if (above + hist[digit] >= remaining) {
          break;
        }
        above += hist[digit];
      }
      s_remaining = remaining - above;
      s_prefix = prefix | ((unsigned int)digit << shift);
      s_mask = mask | (255u << shift);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s_gt = 0u;
    s_eq = 0u;
  }
  __syncthreads();
  const unsigned int threshold = s_prefix;
  const unsigned int want_eq = s_remaining;
  const unsigned int n_gt = topk - want_eq;
  for (int j = threadIdx.x; j < nb; j += blockDim.x) {
    const unsigned int key = q4_ordered(row[j]);
    if (key > threshold) {
      out[atomicAdd(&s_gt, 1u)] = j;
    } else if (key == threshold) {
      const unsigned int e = atomicAdd(&s_eq, 1u);
      if (e < want_eq) {
        out[n_gt + e] = j;
      }
    }
  }
  if (threadIdx.x == 0) {
    n_selected[t] = topk;
  }
}

// Sparse GQA attention of each query over its selected blocks plus the
// incomplete tail. Grid: (tokens, kv_heads, splits); each split covers a
// contiguous chunk of the token's item list and writes partial (max, sum, acc)
// when splits > 1. Head dim is fixed at 256: each lane owns 8 dims.
// flashinfer_layout: K and V are [blocks, kv_heads, block_size, 256]; otherwise
// K is [blocks, kv_heads, 256/8, block_size, 8] and V is [blocks, kv_heads,
// 256, block_size].
template <typename T, typename TC>
__global__ void __launch_bounds__(Q4_ATTN_WARPS * 32) q4_qsa_attention_kernel(
    const T *__restrict__ q, const TC *__restrict__ k_cache,
    const TC *__restrict__ v_cache, Q4Tokens l, Q4Paged pg,
    const int *__restrict__ selected, const int *__restrict__ n_selected,
    int topk, int ratio, int n_q_heads, int n_kv_heads, int items_per_split,
    float scale, int flashinfer_layout, float k_scale, float v_scale, T *__restrict__ out,
    float *__restrict__ part_acc, float *__restrict__ part_ml,
    const T *__restrict__ gate, int64_t gate_tok_stride, int gate_head_stride,
    Q4Q8Block *__restrict__ q8_out) {
  const int t = blockIdx.x;
  const int g = blockIdx.y;
  const int split = blockIdx.z;
  const int splits = gridDim.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = n_q_heads / n_kv_heads;

  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  int n_block_items = 0;
  int n_items = 0;
  int tail_start = 0;
  if (pos >= 0) {
    const int nb = (pos + 1) / ratio;
    n_block_items = n_selected[t] * ratio;
    tail_start = nb * ratio;
    n_items = n_block_items + (pos + 1 - tail_start);
  }
  const int begin = split * items_per_split;
  const int end = min(n_items, begin + items_per_split);

  float qv[Q4_ATTN_MAX_HEADS_PER_WARP][8];
  float acc[Q4_ATTN_MAX_HEADS_PER_WARP][8];
  float m[Q4_ATTN_MAX_HEADS_PER_WARP];
  float s[Q4_ATTN_MAX_HEADS_PER_WARP];
  int n_heads_here = 0;
#pragma unroll
  for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
    const int h = warp + hh * Q4_ATTN_WARPS;
    m[hh] = -FLT_MAX;
    s[hh] = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[hh][i] = 0.0f;
      qv[hh][i] = 0.0f;
    }
    if (h < group) {
      n_heads_here = hh + 1;
      const int qh = g * group + h;
      q4_load8(q + ((size_t)t * n_q_heads + qh) * Q4_ATTN_HEAD_DIM + lane * 8,
               qv[hh]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qv[hh][i] *= scale;
      }
    }
  }

  const size_t head_stride_fi = (size_t)pg.block_size * Q4_ATTN_HEAD_DIM;
  constexpr int X = Q4CacheX<TC>::value;
  for (int it0 = begin; it0 < end; it0 += Q4_ATTN_ITEMS) {
    // Resolve and load a batch of items up front so the dependent loads overlap
    const TC *kp[Q4_ATTN_ITEMS];
    const TC *vp[Q4_ATTN_ITEMS];
#pragma unroll
    for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
      const int it = min(it0 + j, end - 1);
      int p;
      if (it < n_block_items) {
        p = selected[(size_t)t * topk + it / ratio] * ratio + it % ratio;
      } else {
        p = tail_start + (it - n_block_items);
      }
      const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq +
                                      p / pg.block_size];
      const int off = p % pg.block_size;
      const size_t head_base = (size_t)blk * n_kv_heads + g;
      if (flashinfer_layout) {
        const size_t base = head_base * head_stride_fi +
                            (size_t)off * Q4_ATTN_HEAD_DIM + lane * 8;
        kp[j] = k_cache + base;
        vp[j] = v_cache + base;
      } else {
        // K: [blocks, kv_heads, 256/X, block_size, X]; the lane's eight elements
        // d = lane * 8 .. +8 lie in one X-wide column group.
        kp[j] = k_cache +
                (head_base * (Q4_ATTN_HEAD_DIM / X) + (lane * 8) / X) * pg.block_size * X +
                (size_t)off * X + (lane * 8) % X;
        vp[j] = v_cache +
                (head_base * Q4_ATTN_HEAD_DIM + lane * 8) * pg.block_size + off;
      }
    }
    float kf[Q4_ATTN_ITEMS][8];
    float vf[Q4_ATTN_ITEMS][8];
#pragma unroll
    for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
      q4_load8c<TC>(kp[j], k_scale, kf[j]);
      if (flashinfer_layout) {
        q4_load8c<TC>(vp[j], v_scale, vf[j]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          vf[j][i] = q4_cache_f<TC>(vp[j][(size_t)i * pg.block_size], v_scale);
        }
      }
    }
#pragma unroll
    for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
      if (hh < n_heads_here) {
        float dot[Q4_ATTN_ITEMS];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          dot[j] = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            dot[j] = fmaf(qv[hh][i], kf[j][i], dot[j]);
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
          for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
            dot[j] += __shfl_xor_sync(0xffffffff, dot[j], o);
          }
        }
        float m_new = m[hh];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          if (it0 + j < end) {
            m_new = fmaxf(m_new, dot[j]);
          }
        }
        const float corr = __expf(m[hh] - m_new);
        float psum = 0.0f;
        float pj[Q4_ATTN_ITEMS];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          pj[j] = it0 + j < end ? __expf(dot[j] - m_new) : 0.0f;
          psum += pj[j];
        }
        s[hh] = s[hh] * corr + psum;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float a = acc[hh][i] * corr;
#pragma unroll
          for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
            a = fmaf(pj[j], vf[j][i], a);
          }
          acc[hh][i] = a;
        }
        m[hh] = m_new;
      }
    }
  }

#pragma unroll
  for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
    if (hh >= n_heads_here) {
      continue;
    }
    const int qh = g * (n_q_heads / n_kv_heads) + warp + hh * Q4_ATTN_WARPS;
    if (splits == 1) {
      const float inv = s[hh] > 0.0f ? 1.0f / s[hh] : 0.0f;
      T *o = out + ((size_t)t * n_q_heads + qh) * Q4_ATTN_HEAD_DIM + lane * 8;
      // The attention output gate (sigmoid of the q projection's gate half), when fused.
      const T *gt = gate != nullptr
                        ? gate + (size_t)t * gate_tok_stride + (size_t)qh * gate_head_stride + lane * 8
                        : nullptr;
      float ov[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float v = acc[hh][i] * inv;
        const T r = q4_t<T>(gt != nullptr ? v * q4_sigmoid(q4_f(gt[i])) : v);
        o[i] = r;
        ov[i] = q4_f(r);
      }
      if (q8_out != nullptr) {
        // The row's q8_1 form for the output projection: lanes 4b..4b+3 hold block b.
        float amax = 0.0f, sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          amax = fmaxf(amax, fabsf(ov[i]));
          sum += ov[i];
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 32));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, 1, 32);
        sum += __shfl_xor_sync(0xffffffff, sum, 2, 32);
        const float d = amax / 127.0f;
        Q4Q8Block *blk = q8_out + ((size_t)t * n_q_heads + qh) * (Q4_ATTN_HEAD_DIM / 32) + lane / 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          blk->qs[(lane % 4) * 8 + i] = amax == 0.0f ? 0 : (int8_t)roundf(ov[i] / d);
        }
        if (lane % 4 == 0) {
          blk->d = (__half)d;
          blk->s = (__half)sum;
        }
      }
    } else {
      const size_t row = ((size_t)t * n_q_heads + qh) * splits + split;
      float *a = part_acc + row * Q4_ATTN_HEAD_DIM + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        a[i] = acc[hh][i];
      }
      if (lane == 0) {
        part_ml[row * 2] = m[hh];
        part_ml[row * 2 + 1] = s[hh];
      }
    }
  }
}

template <typename T>
__global__ void
q4_qsa_attention_reduce_kernel(const float *__restrict__ part_acc,
                               const float *__restrict__ part_ml, int splits,
                               T *__restrict__ out, int n_q_heads,
                               const T *__restrict__ gate, int64_t gate_tok_stride,
                               int gate_head_stride, Q4Q8Block *__restrict__ q8_out) {
  const size_t row = blockIdx.x;
  const int d = threadIdx.x;
  float m = -FLT_MAX;
  for (int sp = 0; sp < splits; ++sp) {
    const float ms = part_ml[(row * splits + sp) * 2];
    if (part_ml[(row * splits + sp) * 2 + 1] > 0.0f) {
      m = fmaxf(m, ms);
    }
  }
  float total = 0.0f;
  float acc = 0.0f;
  for (int sp = 0; sp < splits; ++sp) {
    const float ss = part_ml[(row * splits + sp) * 2 + 1];
    if (ss <= 0.0f) {
      continue;
    }
    const float w = __expf(part_ml[(row * splits + sp) * 2] - m);
    total = fmaf(ss, w, total);
    acc = fmaf(part_acc[(row * splits + sp) * Q4_ATTN_HEAD_DIM + d], w, acc);
  }
  float v = total > 0.0f ? acc / total : 0.0f;
  if (gate != nullptr) {
    const size_t t = row / n_q_heads, qh = row - t * n_q_heads;
    v *= q4_sigmoid(q4_f(gate[t * gate_tok_stride + qh * gate_head_stride + d]));
  }
  const T r = q4_t<T>(v);
  out[row * Q4_ATTN_HEAD_DIM + d] = r;
  if (q8_out != nullptr) {
    // Each wave of 32 threads holds one q8_1 block of the row.
    const float ov = q4_f(r);
    float amax = fabsf(ov), sum = ov;
    for (int off = 16; off > 0; off >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
      sum += __shfl_xor_sync(0xffffffff, sum, off, 32);
    }
    const float dq = amax / 127.0f;
    Q4Q8Block *blk = q8_out + row * (Q4_ATTN_HEAD_DIM / 32) + d / 32;
    blk->qs[d % 32] = amax == 0.0f ? 0 : (int8_t)roundf(ov / dq);
    if (d % 32 == 0) {
      blk->d = (__half)dq;
      blk->s = (__half)sum;
    }
  }
}

// Launcher dtype codes: 0 = f16, 1 = bf16, 2 = f32.

#define Q4_DISPATCH(dtype, ...)                                                \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else if ((dtype) == 1) {                                                 \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = float;                                                  \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)


#ifdef USE_ROCM
// Prefill variant of q4_qsa_attention_kernel for RDNA (wave32 WMMA). One
// workgroup of eight waves per (token, kv head): the query heads sharing that kv
// head form the rows of a 16-row tile, and the token's selected items are consumed
// 32 at a time with an online softmax. Scores and P.V run on WMMA through rocWMMA.
// The gather is software pipelined: while a step's tile is multiplied, the next
// step's K/V rows are already in flight into registers (the kernel was 65% load
// waits at two waves per SIMD without it). Head dim 256, at most 16 query heads per
// kv head.
#include <rocwmma/rocwmma.hpp>

#define Q4W_ROWS 16
#define Q4W_ITEMS 32
#define Q4W_LD (Q4_ATTN_HEAD_DIM + 8)
#define Q4W_THREADS 256
// Eight-element pieces of K and of V per step, per thread.
#define Q4W_PIECES (Q4W_ITEMS * (Q4_ATTN_HEAD_DIM / 8) / Q4W_THREADS)

template <typename TC> struct Q4WRaw8 {
  TC x[8];
};

template <typename T, typename TC>
__global__ void __launch_bounds__(Q4W_THREADS) q4_qsa_attention_wmma_kernel(
    const T *__restrict__ q, const TC *__restrict__ k_cache,
    const TC *__restrict__ v_cache, Q4Tokens l, Q4Paged pg,
    const int *__restrict__ selected, const int *__restrict__ n_selected,
    int topk, int ratio, int n_q_heads, int n_kv_heads, float scale,
    int flashinfer_layout, float k_scale, float v_scale, T *__restrict__ out) {
  using namespace rocwmma;
  constexpr int X = Q4CacheX<TC>::value;
  constexpr int GROUPS = Q4_ATTN_HEAD_DIM / 8;
  // fp16 operands: after the q/k norms the scores, probabilities and values are small,
  // and fp16's mantissa keeps the products ~8x closer to the f32 kernel than bf16.
  using bf = rocwmma::float16_t;
  __shared__ bf q_lds[Q4W_ROWS * Q4W_LD];
  __shared__ bf k_lds[Q4W_ITEMS * Q4W_LD];
  __shared__ bf v_lds[Q4W_ITEMS * Q4W_LD];
  __shared__ float s_part[4][Q4W_ROWS * Q4W_ITEMS];
  __shared__ bf p_lds[Q4W_ROWS * (Q4W_ITEMS + 8)];
  __shared__ float m_row[Q4W_ROWS], l_row[Q4W_ROWS], corr_row[Q4W_ROWS];
  // Item row pointers of the step being loaded (double buffered against the step
  // being converted).
  __shared__ const TC *k_item[2][Q4W_ITEMS];
  __shared__ const TC *v_item[2][Q4W_ITEMS];

  const int t = blockIdx.x;
  const int g = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int group = n_q_heads / n_kv_heads;

  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  int n_block_items = 0, n_items = 0, tail_start = 0;
  if (pos >= 0) {
    const int nb = (pos + 1) / ratio;
    n_block_items = n_selected[t] * ratio;
    tail_start = nb * ratio;
    n_items = n_block_items + (pos + 1 - tail_start);
  }
  const size_t head_stride_fi = (size_t)pg.block_size * Q4_ATTN_HEAD_DIM;

  // Row pointers of items it0 .. it0 + 31 into buffer `buf` (threads 0..31).
  auto resolve = [&](const int it0, const int buf) {
    if (tid < Q4W_ITEMS) {
      const int it = it0 + tid;
      const TC *kp = nullptr, *vp = nullptr;
      if (it < n_items) {
        const int p = it < n_block_items
                          ? selected[(size_t)t * topk + it / ratio] * ratio + it % ratio
                          : tail_start + (it - n_block_items);
        const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq + p / pg.block_size];
        const int off = p % pg.block_size;
        const size_t head_base = (size_t)blk * n_kv_heads + g;
        if (flashinfer_layout) {
          kp = k_cache + head_base * head_stride_fi + (size_t)off * Q4_ATTN_HEAD_DIM;
          vp = v_cache + head_base * head_stride_fi + (size_t)off * Q4_ATTN_HEAD_DIM;
        } else {
          // K: [blocks, kv_heads, 256/X, block_size, X]; V: [blocks, kv_heads, 256, block_size].
          kp = k_cache + head_base * Q4_ATTN_HEAD_DIM * pg.block_size + (size_t)off * X;
          vp = v_cache + head_base * Q4_ATTN_HEAD_DIM * pg.block_size + off;
        }
      }
      k_item[buf][tid] = kp;
      v_item[buf][tid] = vp;
    }
  };
  // The thread's pieces: piece i covers item j = e / GROUPS, element group c = e % GROUPS.
  auto piece = [&](const int i, int &j, int &c) {
    const int e = tid + i * Q4W_THREADS;
    j = e / GROUPS;
    c = e % GROUPS;
  };
  // Global loads of one step's K and V pieces into registers (no conversion yet).
  Q4WRaw8<TC> kraw[Q4W_PIECES], vraw[Q4W_PIECES];
  auto load_step = [&](const int buf) {
#pragma unroll
    for (int i = 0; i < Q4W_PIECES; ++i) {
      int j, c;
      piece(i, j, c);
      const TC *kp = k_item[buf][j];
      const TC *vp = v_item[buf][j];
      if (kp) {
        if (flashinfer_layout) {
          kraw[i] = *reinterpret_cast<const Q4WRaw8<TC> *>(kp + c * 8);
          vraw[i] = *reinterpret_cast<const Q4WRaw8<TC> *>(vp + c * 8);
        } else {
          // K's eight elements sit in one X-wide column group; V's are block_size apart.
          kraw[i] = *reinterpret_cast<const Q4WRaw8<TC> *>(
              kp + (size_t)((c * 8) / X) * pg.block_size * X + (c * 8) % X);
#pragma unroll
          for (int u = 0; u < 8; ++u) vraw[i].x[u] = vp[(size_t)(c * 8 + u) * pg.block_size];
        }
      } else {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          kraw[i].x[u] = TC{};
          vraw[i].x[u] = TC{};
        }
      }
    }
  };
  // The registers into the K/V tiles as fp16.
  auto store_step = [&]() {
#pragma unroll
    for (int i = 0; i < Q4W_PIECES; ++i) {
      int j, c;
      piece(i, j, c);
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        k_lds[j * Q4W_LD + c * 8 + u] = static_cast<bf>(q4_cache_f<TC>(kraw[i].x[u], k_scale));
        v_lds[j * Q4W_LD + c * 8 + u] = static_cast<bf>(q4_cache_f<TC>(vraw[i].x[u], v_scale));
      }
    }
  };

  for (int e = tid; e < Q4W_ROWS * Q4_ATTN_HEAD_DIM; e += Q4W_THREADS) {
    const int r = e / Q4_ATTN_HEAD_DIM, d = e % Q4_ATTN_HEAD_DIM;
    float v = 0.0f;
    if (r < group) {
      v = q4_f(q[((size_t)t * n_q_heads + g * group + r) * Q4_ATTN_HEAD_DIM + d]) * scale;
    }
    q_lds[r * Q4W_LD + d] = static_cast<bf>(v);
  }
  if (tid < Q4W_ROWS) {
    m_row[tid] = -FLT_MAX;
    l_row[tid] = 0.0f;
  }
  // Each wave accumulates 32 of the 256 output dims in registers: on gfx11/12 lane
  // `lane` holds rows i + 8 * (lane / 16), column lane % 16, of each 16 x 16 tile.
  fragment<accumulator, 16, 16, 16, float32_t> o_acc[2];
  fill_fragment(o_acc[0], 0.0f);
  fill_fragment(o_acc[1], 0.0f);

  // Prologue: the first step's pointers and loads.
  resolve(0, 0);
  __syncthreads();
  if (n_items > 0) {
    load_step(0);
  }

  int buf = 0;
  for (int it0 = 0; it0 < n_items; it0 += Q4W_ITEMS) {
    const bool more = it0 + Q4W_ITEMS < n_items;
    // The previous step's P.V may still read v_lds.
    __syncthreads();
    if (more) {
      resolve(it0 + Q4W_ITEMS, buf ^ 1);
    }
    store_step();
    __syncthreads();
    // The next step's rows start flying now and land during the MMAs below.
    if (more) {
      load_step(buf ^ 1);
    }
    // S = Q K^T: wave (kq, h) multiplies dims kq * 64 .. +64 against items h * 16 .. +16.
    {
      const int kq = wave % 4, h = wave / 4;
      fragment<accumulator, 16, 16, 16, float32_t> sacc;
      fill_fragment(sacc, 0.0f);
#pragma unroll
      for (int kk = kq * 64; kk < kq * 64 + 64; kk += 16) {
        fragment<matrix_a, 16, 16, 16, bf, row_major> a;
        fragment<matrix_b, 16, 16, 16, bf, col_major> b;
        load_matrix_sync(a, q_lds + kk, Q4W_LD);
        load_matrix_sync(b, k_lds + (h * 16) * Q4W_LD + kk, Q4W_LD);
        mma_sync(sacc, a, b, sacc);
      }
      store_matrix_sync(s_part[kq] + h * 16, sacc, Q4W_ITEMS, mem_row_major);
    }
    __syncthreads();
    // Online softmax: sixteen threads per row, two items each.
    {
      const int r = tid / 16, sub = tid % 16;
      float sv[2];
      float mx = -FLT_MAX;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int j = sub * 2 + u;
        float v = s_part[0][r * Q4W_ITEMS + j] + s_part[1][r * Q4W_ITEMS + j] +
                  s_part[2][r * Q4W_ITEMS + j] + s_part[3][r * Q4W_ITEMS + j];
        if (it0 + j >= n_items) v = -FLT_MAX;
        sv[u] = v;
        mx = fmaxf(mx, v);
      }
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o, 16));
      const float m_old = m_row[r];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int j = sub * 2 + u;
        const float pv = it0 + j < n_items ? __expf(sv[u] - m_new) : 0.0f;
        sum += pv;
        p_lds[r * (Q4W_ITEMS + 8) + j] = static_cast<bf>(pv);
      }
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o, 16);
      if (sub == 0) {
        const float corr = __expf(m_old - m_new);
        corr_row[r] = corr;
        l_row[r] = l_row[r] * corr + sum;
        m_row[r] = m_new;
      }
    }
    __syncthreads();
    // Rescale the output rows in registers, then O += P V over this wave's two column tiles.
    {
      float corr[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) corr[i] = corr_row[i + 8 * (lane / 16)];
#pragma unroll
      for (int c = 0; c < 2; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o_acc[c].x[i] *= corr[i];
      }
#pragma unroll
      for (int kk = 0; kk < Q4W_ITEMS; kk += 16) {
        fragment<matrix_a, 16, 16, 16, bf, row_major> a;
        load_matrix_sync(a, p_lds + kk, Q4W_ITEMS + 8);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          fragment<matrix_b, 16, 16, 16, bf, row_major> b;
          load_matrix_sync(b, v_lds + kk * Q4W_LD + (wave * 2 + c) * 16, Q4W_LD);
          mma_sync(o_acc[c], a, b, o_acc[c]);
        }
      }
    }
    buf ^= 1;
  }
  __syncthreads();
#pragma unroll
  for (int c = 0; c < 2; ++c) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = i + 8 * (lane / 16);
      const int d = (wave * 2 + c) * 16 + lane % 16;
      if (r < group) {
        const float inv = l_row[r] > 0.0f ? 1.0f / l_row[r] : 0.0f;
        out[((size_t)t * n_q_heads + g * group + r) * Q4_ATTN_HEAD_DIM + d] =
            q4_t<T>(o_acc[c].x[i] * inv);
      }
    }
  }
}
#endif

#define Q4_DISPATCH_HALF(dtype, ...)                                           \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)

static inline int q4_row_threads(int hidden) {
  int threads = 32;
  while (threads < 512 && threads * 4 < hidden) {
    threads <<= 1;
  }
  return threads;
}

extern "C" void qwen4_hc_norm(const void *x, const float *weight, void *out,
                              int rows, int hc, int hidden, float eps,
                              int dtype, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_hc_norm_kernel<scalar_t>
              <<<rows, q4_row_threads(hidden), 0, s>>>((const scalar_t *)x,
                                                       weight, (scalar_t *)out,
                                                       hc, hidden, eps));
}

extern "C" void qwen4_hc_mix(const void *xn, const void *gate, void *out,
                             int tokens, int hc, int hidden, int dtype,
                             int64_t stream) {
  const size_t total = (size_t)tokens * hidden;
  if (total == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 256;
  const unsigned int blocks = (total + threads - 1) / threads;
  Q4_DISPATCH(dtype, q4_hc_mix_kernel<scalar_t><<<blocks, threads, 0, s>>>(
                         (const scalar_t *)xn, (const scalar_t *)gate,
                         (scalar_t *)out, total, hc, hidden));
}

extern "C" void qwen4_hc_combine(const void *res, const void *block_out,
                                 const void *inject, void *res_out,
                                 const float *next_weight, void *xn_out,
                                 int rows, int hc, int hidden, float eps,
                                 int dtype, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_hc_combine_kernel<scalar_t>
              <<<rows, q4_row_threads(hidden), 0, s>>>(
                  (const scalar_t *)res, (const scalar_t *)block_out,
                  (const scalar_t *)inject, (scalar_t *)res_out, next_weight,
                  (scalar_t *)xn_out, hc, hidden, eps));
}

#ifdef USE_ROCM
// `peers`, `rank`, `world`, `max_elems`, `epoch`: a candle PeerAllReduce's raw handles;
// `finished`: its zeroed workgroup counter. `hidden / hc` must be a multiple of
// 16 bytes of T.
extern "C" void qwen4_hc_combine_ar(const void *res, const void *block_out,
                                    const void *inject, void *res_out,
                                    const float *next_weight, void *xn_out,
                                    int rows, int hc, int hidden, float eps,
                                    int dtype, const uint64_t *peers, int rank,
                                    int world, int max_elems, int *epoch,
                                    unsigned int *finished, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_hc_combine_ar_kernel<scalar_t>
              <<<rows, q4_row_threads(hidden), 0, s>>>(
                  (const scalar_t *)res, (const scalar_t *)block_out,
                  (const scalar_t *)inject, (scalar_t *)res_out, next_weight,
                  (scalar_t *)xn_out, hc, hidden, eps, peers, rank, world,
                  max_elems, epoch, finished));
}
#endif

extern "C" void qwen4_index_q_prep(const void *index_qk, int in_stride, const void *norm_w,
                                   float eps, const void *cos, const void *sin, int half_rot,
                                   void *index_q, void *raw_key, int tokens, int heads, int d,
                                   int dtype, int64_t stream) {
  if (tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const dim3 grid(tokens, heads);
  Q4_DISPATCH(dtype, q4_index_q_prep_kernel<scalar_t><<<grid, d, 0, s>>>(
                         (const scalar_t *)index_qk, in_stride, (const scalar_t *)norm_w, eps,
                         (const scalar_t *)cos, (const scalar_t *)sin, half_rot,
                         (scalar_t *)index_q, (scalar_t *)raw_key, heads, d));
}

extern "C" void qwen4_ple_hash(const unsigned int *tokens, Q4Tokens layout,
                               const float *hist_pool,
                               const unsigned int *slots, int hist_len,
                               const Q4PleHashParams *params,
                               long long *rows_out, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 128;
  q4_ple_hash_kernel<<<(layout.n_tokens + threads - 1) / threads, threads, 0,
                       s>>>(tokens, layout, hist_pool, slots, hist_len, *params,
                            rows_out);
}

extern "C" void qwen4_ple_hist_update(const unsigned int *tokens,
                                      Q4Tokens layout, float *hist_pool,
                                      const unsigned int *slots, int hist_len,
                                      int64_t stream) {
  if (layout.n_seqs == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  q4_ple_hist_update_kernel<<<(layout.n_seqs + 63) / 64, 64, 0, s>>>(
      tokens, layout, hist_pool, slots, hist_len);
}

extern "C" void qwen4_ple_gather(const long long *rows, int n_lookups,
                                 const unsigned long long *shard_ptrs,
                                 const long long *shard_starts, int n_shards,
                                 int head_dim, void *out, int64_t stream) {
  if (n_lookups == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  q4_ple_gather_kernel<<<(n_lookups + warps - 1) / warps, warps * 32, 0, s>>>(
      rows, n_lookups, shard_ptrs, shard_starts, n_shards, head_dim,
      (unsigned short *)out);
}

extern "C" void qwen4_ple_gather_quant(const long long *rows, int n_lookups,
                                       const unsigned char *data,
                                       const void *scales, int head_dim,
                                       int format, void *out, int64_t stream) {
  if (n_lookups == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  q4_ple_gather_quant_kernel<<<(n_lookups + warps - 1) / warps, warps * 32, 0,
                               s>>>(rows, n_lookups, data,
                                    (const __half *)scales, head_dim, format,
                                    (__nv_bfloat16 *)out);
}

extern "C" void qwen4_ple_gate(const void *key, const void *hidden_states,
                               const void *value, const float *w_key,
                               const float *w_query, const float *w_conv,
                               void *gated_out, void *normed_out, int tokens,
                               int hc, int hidden, float eps, int dtype,
                               int64_t stream) {
  if (tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_ple_gate_kernel<scalar_t><<<tokens, 512, 0, s>>>(
                         (const scalar_t *)key, (const scalar_t *)hidden_states,
                         (const scalar_t *)value, w_key, w_query, w_conv,
                         (scalar_t *)gated_out, (scalar_t *)normed_out, hc,
                         hidden, eps));
}

extern "C" void qwen4_ple_conv(const void *normed, const void *gated,
                               const void *residual, const void *weight,
                               void *state_pool, const unsigned int *slots,
                               void *out, Q4Tokens layout, int channels,
                               int kernel_size, int dilation, int state_len,
                               int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 256;
  const size_t total = (size_t)layout.n_tokens * channels;
  const unsigned int blocks = (total + threads - 1) / threads;
  const size_t state_total = (size_t)layout.n_seqs * channels;
  const unsigned int state_blocks = (state_total + threads - 1) / threads;
  Q4_DISPATCH(dtype, {
    q4_ple_conv_kernel<scalar_t><<<blocks, threads, 0, s>>>(
        (const scalar_t *)normed, (const scalar_t *)gated,
        (const scalar_t *)residual, (const scalar_t *)weight,
        (const scalar_t *)state_pool, slots, (scalar_t *)out, layout, channels,
        kernel_size, dilation, state_len);
    q4_ple_conv_state_update_kernel<scalar_t><<<state_blocks, threads, 0, s>>>(
        (const scalar_t *)normed, (scalar_t *)state_pool, slots, layout,
        channels, state_len);
  });
}

extern "C" void qwen4_qsa_aux_write(const void *raw_key, const void *cos,
                                    const void *sin,
                                    const long long *slot_mapping, void *aux,
                                    void *block_cs, int n_tokens, int key_dim,
                                    int half_rot, int aux_dim, int ratio,
                                    int dtype, int64_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH_HALF(dtype,
                   q4_qsa_aux_write_kernel<scalar_t><<<n_tokens, 128, 0, s>>>(
                       (const scalar_t *)raw_key, (const scalar_t *)cos,
                       (const scalar_t *)sin, slot_mapping, (uint8_t *)aux,
                       (scalar_t *)block_cs, n_tokens, key_dim, half_rot,
                       aux_dim, ratio));
}

extern "C" void qwen4_qsa_finalize(const void *aux, const void *carry,
                                   const unsigned *state_slots, int lanes,
                                   const void *block_cs, void *block_keys,
                                   Q4Tokens layout, Q4Paged paged,
                                   const float *norm_weight, int ratio, int half_rot,
                                   int aux_dim, float eps, int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  Q4_DISPATCH_HALF(
      dtype, q4_qsa_finalize_kernel<scalar_t>
      <<<(layout.n_tokens + warps - 1) / warps, warps * 32, 0, s>>>(
          (const uint8_t *)aux, (const uint8_t *)carry, state_slots, lanes,
          (const scalar_t *)block_cs, (scalar_t *)block_keys, layout, paged,
          norm_weight, ratio, half_rot, aux_dim, eps));
  // Stream order keeps the carry update behind the finalize that reads the same scratch.
  q4_qsa_carry_update_kernel<<<layout.n_seqs, 128, 0, s>>>(
      (const uint8_t *)aux, (uint8_t *)carry, state_slots, lanes, layout, paged, ratio,
      aux_dim);
}

extern "C" void qwen4_qsa_select(const void *q, const void *block_keys,
                                 Q4Tokens layout, Q4Paged paged, float *scores,
                                 int score_stride, int n_heads, int ratio,
                                 int topk, int key_dim, int *selected,
                                 int *n_selected, int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  if (score_stride > topk) {
    const int threads = 128;
    const dim3 grid((score_stride + threads - 1) / threads, layout.n_tokens);
    Q4_DISPATCH_HALF(
        dtype, q4_qsa_score_kernel<scalar_t><<<grid, threads, 0, s>>>(
                   (const scalar_t *)q, (const scalar_t *)block_keys, layout,
                   paged, scores, score_stride, n_heads, ratio, topk, key_dim));
  }
  q4_qsa_topk_kernel<<<layout.n_tokens, 1024, 0, s>>>(
      scores, score_stride, layout, paged, ratio, topk, selected, n_selected);
}

// Top-k over scores produced elsewhere (the cuTile scoring kernel).
extern "C" void qwen4_qsa_topk(const float *scores, int score_stride,
                               Q4Tokens layout, Q4Paged paged, int ratio,
                               int topk, int *selected, int *n_selected,
                               int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  q4_qsa_topk_kernel<<<layout.n_tokens, 1024, 0, (cudaStream_t)stream>>>(
      scores, score_stride, layout, paged, ratio, topk, selected, n_selected);
}

// `gate` (nullable): the q projection's gate half, [tokens] x [heads] strided as given; the
// kernels multiply the output by its sigmoid. The WMMA prefill path ignores it, so callers
// pass it only for decode-sized batches (fewer than 64 tokens).
extern "C" void qwen4_qsa_attention(
    const void *q, const void *k_cache, const void *v_cache, Q4Tokens layout,
    Q4Paged paged, const int *selected, const int *n_selected, int topk,
    int ratio, int n_q_heads, int n_kv_heads, int splits, int items_per_split,
    float scale, int flashinfer_layout, void *out, float *part_acc,
    float *part_ml, const void *gate, int64_t gate_tok_stride,
    int gate_head_stride, void *q8_out, int dtype, int cache_fp8, float k_scale,
    float v_scale, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  // `cache_fp8` selects an OCP e4m3 cache (elements decoded with k_scale / v_scale);
  // otherwise the cache holds the activation type.
#define Q4_QSA_WITH_CACHE_T(...)                                               \
  do {                                                                         \
    if (cache_fp8) {                                                           \
      using cache_t = uint8_t;                                                 \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using cache_t = scalar_t;                                                \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)
#ifdef USE_ROCM
  // Prefill batches take the WMMA kernel; decode keeps the split kernel below.
  if (gate == nullptr && q8_out == nullptr && layout.n_tokens >= 64 && n_q_heads % n_kv_heads == 0 &&
      n_q_heads / n_kv_heads <= Q4W_ROWS) {
    const dim3 wgrid(layout.n_tokens, n_kv_heads);
    Q4_DISPATCH_HALF(dtype, {
      Q4_QSA_WITH_CACHE_T({
        q4_qsa_attention_wmma_kernel<scalar_t, cache_t><<<wgrid, Q4W_THREADS, 0, s>>>(
            (const scalar_t *)q, (const cache_t *)k_cache,
            (const cache_t *)v_cache, layout, paged, selected, n_selected, topk,
            ratio, n_q_heads, n_kv_heads, scale, flashinfer_layout, k_scale, v_scale,
            (scalar_t *)out);
      });
    });
    return;
  }
#endif
  const dim3 grid(layout.n_tokens, n_kv_heads, splits);
  Q4_DISPATCH_HALF(dtype, {
    Q4_QSA_WITH_CACHE_T({
      q4_qsa_attention_kernel<scalar_t, cache_t><<<grid, Q4_ATTN_WARPS * 32, 0, s>>>(
          (const scalar_t *)q, (const cache_t *)k_cache,
          (const cache_t *)v_cache, layout, paged, selected, n_selected, topk,
          ratio, n_q_heads, n_kv_heads, items_per_split, scale, flashinfer_layout,
          k_scale, v_scale, (scalar_t *)out, part_acc, part_ml, (const scalar_t *)gate,
          gate_tok_stride, gate_head_stride, (Q4Q8Block *)q8_out);
    });
    if (splits > 1) {
      q4_qsa_attention_reduce_kernel<scalar_t>
          <<<layout.n_tokens * n_q_heads, Q4_ATTN_HEAD_DIM, 0, s>>>(
              part_acc, part_ml, splits, (scalar_t *)out, n_q_heads,
              (const scalar_t *)gate, gate_tok_stride, gate_head_stride,
              (Q4Q8Block *)q8_out);
    }
  });
}

#ifdef USE_ROCM
// Dense mode on a token-major cache: every complete block of each query is selected (the
// paged-attention layer's standard kernels only know the vLLM layout).
__global__ void q4_qsa_select_all_kernel(Q4Tokens l, Q4Paged pg, int ratio, int topk,
                                         int *__restrict__ selected,
                                         int *__restrict__ n_selected) {
  const int t = blockIdx.x;
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  const int nb = pos < 0 ? 0 : min((pos + 1) / ratio, topk);
  for (int j = threadIdx.x; j < nb; j += blockDim.x) {
    selected[(size_t)t * topk + j] = j;
  }
  if (threadIdx.x == 0) {
    n_selected[t] = nb;
  }
}

extern "C" void qwen4_qsa_select_all(Q4Tokens layout, Q4Paged paged, int ratio, int topk,
                                     int *selected, int *n_selected, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  q4_qsa_select_all_kernel<<<layout.n_tokens, 128, 0, (cudaStream_t)stream>>>(
      layout, paged, ratio, topk, selected, n_selected);
}

// A cache element from an activation: fp8 caches store x / scale (what the attention
// kernels multiply back by `k_scale` / `v_scale`).
template <typename T, typename TC>
__device__ __forceinline__ TC q4_cache_store(T x, float inv_scale) {
  return q4_t<TC>(q4_f(x));
}
template <>
__device__ __forceinline__ uint8_t q4_cache_store<__half, uint8_t>(__half x, float inv_scale) {
  return __hip_cvt_float_to_fp8(q4_f(x) * inv_scale, __HIP_SATFINITE, __HIP_E4M3);
}
template <>
__device__ __forceinline__ uint8_t q4_cache_store<__nv_bfloat16, uint8_t>(__nv_bfloat16 x,
                                                                            float inv_scale) {
  return __hip_cvt_float_to_fp8(q4_f(x) * inv_scale, __HIP_SATFINITE, __HIP_E4M3);
}

// Writes `k`, `v` ([tokens, kv_heads, head_dim], contiguous) into a token-major paged cache
// ([blocks, kv_heads, block_size, head_dim]) at `slots` (block * block_size + offset; a
// negative slot skips the token). Eight consecutive elements per thread.
template <typename T, typename TC>
__global__ void q4_kv_write_hnd_kernel(const T *__restrict__ k, const T *__restrict__ v,
                                       const int64_t *__restrict__ slots, TC *__restrict__ kc,
                                       TC *__restrict__ vc, int kv_heads, int head_dim,
                                       int block_size, const float *__restrict__ k_scale,
                                       const float *__restrict__ v_scale) {
  const int t = blockIdx.x;
  const int64_t slot = slots[t];
  if (slot < 0) {
    return;
  }
  const int64_t blk = slot / block_size, off = slot % block_size;
  const float ki = k_scale ? 1.0f / *k_scale : 1.0f;
  const float vi = v_scale ? 1.0f / *v_scale : 1.0f;
  const int n = kv_heads * head_dim;
  for (int e8 = threadIdx.x * 8; e8 < n; e8 += blockDim.x * 8) {
    const int h = e8 / head_dim, d = e8 - h * head_dim;
    const size_t src = (size_t)t * n + e8;
    const size_t dst = (((size_t)blk * kv_heads + h) * block_size + off) * head_dim + d;
    struct __align__(16) In8 {
      T x[8];
    };
    struct __align__(8 * sizeof(TC)) Out8 {
      TC x[8];
    };
    const In8 kin = *reinterpret_cast<const In8 *>(k + src);
    const In8 vin = *reinterpret_cast<const In8 *>(v + src);
    Out8 kout, vout;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      kout.x[i] = q4_cache_store<T, TC>(kin.x[i], ki);
      vout.x[i] = q4_cache_store<T, TC>(vin.x[i], vi);
    }
    *reinterpret_cast<Out8 *>(kc + dst) = kout;
    *reinterpret_cast<Out8 *>(vc + dst) = vout;
  }
}

extern "C" void qwen4_kv_write_hnd(const void *k, const void *v, const int64_t *slots, void *kc,
                                   void *vc, int n_tokens, int kv_heads, int head_dim,
                                   int block_size, const float *k_scale, const float *v_scale,
                                   int dtype, int cache_fp8, int64_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const int threads = min(256, max(32, (kv_heads * head_dim / 8 + 31) / 32 * 32));
  Q4_DISPATCH_HALF(dtype, {
    Q4_QSA_WITH_CACHE_T({
      q4_kv_write_hnd_kernel<scalar_t, cache_t><<<n_tokens, threads, 0, (cudaStream_t)stream>>>(
          (const scalar_t *)k, (const scalar_t *)v, slots, (cache_t *)kc, (cache_t *)vc,
          kv_heads, head_dim, block_size, k_scale, v_scale);
    });
  });
}
#endif
