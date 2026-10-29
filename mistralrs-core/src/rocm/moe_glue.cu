// Elementwise kernels that replace short chains of generic ops on ROCm: the
// gated activation of GLU MLPs, the top-k weighted sum of MoE outputs and the
// partial NeoX RoPE of the Qwen4-Exp indexer.
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

template <typename T> __device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }
template <> __device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T> __device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// Activation codes follow mistralrs_quant::GluActivationType.
__device__ __forceinline__ float glu_activation(float x, int kind) {
  switch (kind) {
  case 0:
    return x / (1.0f + expf(-x));
  case 1:
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
  case 2:
    return fmaxf(x, 0.0f);
  case 3:
    return 0.5f * x * (1.0f + erff(x * 0.7071067812f));
  default:
    return 1.0f / (1.0f + expf(-x));
  }
}

// out = act(gate) * up, all contiguous with n elements.
template <typename T>
__global__ void glu_act_mul_kernel(const T *__restrict__ gate, const T *__restrict__ up,
                                   T *__restrict__ out, const int64_t n, const int kind) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = from_float<T>(glu_activation(to_float(gate[i]), kind) * to_float(up[i]));
  }
}

// out[t, h] = sum_k ys[t, k, h] * weights[t, k], accumulated in f32.
template <typename T>
__global__ void moe_weighted_reduce_kernel(const T *__restrict__ ys, const float *__restrict__ weights,
                                           T *__restrict__ out, const int top_k, const int hidden) {
  const int t = blockIdx.y;
  const T *row = ys + (int64_t)t * top_k * hidden;
  const float *w = weights + (int64_t)t * top_k;
  for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < hidden; h += gridDim.x * blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < top_k; k++) {
      acc += to_float(row[(int64_t)k * hidden + h]) * w[k];
    }
    out[(int64_t)t * hidden + h] = from_float<T>(acc);
  }
}

#define GLUE_OPS(T, SUFFIX)                                                                     \
  extern "C" void glu_act_mul_##SUFFIX(const void *gate, const void *up, void *out, int64_t n, \
                                       int kind, int64_t stream) {                             \
    const int threads = 256;                                                                   \
    const int64_t blocks = (n + threads - 1) / threads;                                        \
    glu_act_mul_kernel<T><<<(unsigned)(blocks < 65535 ? blocks : 65535), threads, 0,           \
                            (cudaStream_t)stream>>>((const T *)gate, (const T *)up, (T *)out,  \
                                                    n, kind);                                  \
  }                                                                                            \
  extern "C" void moe_weighted_reduce_##SUFFIX(const void *ys, const float *weights, void *out, \
                                               int tokens, int top_k, int hidden,              \
                                               int64_t stream) {                               \
    const int threads = 256;                                                                   \
    dim3 grid((hidden + threads - 1) / threads, tokens, 1);                                    \
    moe_weighted_reduce_kernel<T><<<grid, threads, 0, (cudaStream_t)stream>>>(                 \
        (const T *)ys, weights, (T *)out, top_k, hidden);                                      \
  }

GLUE_OPS(float, f32)
GLUE_OPS(__half, f16)
GLUE_OPS(__nv_bfloat16, bf16)

// NeoX RoPE over the first 2 * half dims of each head, in f32. x and out are
// contiguous [rows, heads, d]; cos and sin are f32 [rows, half].
template <typename T>
__global__ void rope_partial_neox_kernel(const T *__restrict__ x, const float *__restrict__ cos,
                                         const float *__restrict__ sin, T *__restrict__ out,
                                         const int64_t n, const int heads, const int d,
                                         const int half) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int c = i % d;
    const int64_t row = i / d / heads;
    const float v = to_float(x[i]);
    float r = v;
    if (c < half) {
      r = v * cos[row * half + c] - to_float(x[i + half]) * sin[row * half + c];
    } else if (c < 2 * half) {
      const int j = c - half;
      r = v * cos[row * half + j] + to_float(x[i - half]) * sin[row * half + j];
    }
    out[i] = from_float<T>(r);
  }
}

#define ROPE_OP(T, SUFFIX)                                                                     \
  extern "C" void rope_partial_neox_##SUFFIX(const void *x, const float *cos, const float *sin, \
                                             void *out, int64_t n, int heads, int d, int half, \
                                             int64_t stream) {                                 \
    const int threads = 256;                                                                   \
    const int64_t blocks = (n + threads - 1) / threads;                                        \
    rope_partial_neox_kernel<T><<<(unsigned)(blocks < 65535 ? blocks : 65535), threads, 0,     \
                                  (cudaStream_t)stream>>>((const T *)x, cos, sin, (T *)out, n, \
                                                          heads, d, half);                     \
  }

ROPE_OP(float, f32)
ROPE_OP(__half, f16)
ROPE_OP(__nv_bfloat16, bf16)

// out[t, h] = y[t, h] + sigmoid(gate[t]) * shared[t, h]: the gated shared
// expert of Qwen MoE blocks. gate holds one logit per token.
template <typename T>
__global__ void add_sigmoid_gated_kernel(const T *__restrict__ y, const T *__restrict__ shared,
                                         const T *__restrict__ gate, T *__restrict__ out,
                                         const int64_t n, const int hidden) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float g = 1.0f / (1.0f + expf(-to_float(gate[i / hidden])));
    out[i] = from_float<T>(to_float(y[i]) + g * to_float(shared[i]));
  }
}

#define GATED_ADD_OP(T, SUFFIX)                                                                \
  extern "C" void add_sigmoid_gated_##SUFFIX(const void *y, const void *shared,                \
                                             const void *gate, void *out, int64_t n,           \
                                             int hidden, int64_t stream) {                     \
    const int threads = 256;                                                                   \
    const int64_t blocks = (n + threads - 1) / threads;                                        \
    add_sigmoid_gated_kernel<T><<<(unsigned)(blocks < 65535 ? blocks : 65535), threads, 0,     \
                                  (cudaStream_t)stream>>>((const T *)y, (const T *)shared,     \
                                                          (const T *)gate, (T *)out, n,        \
                                                          hidden);                             \
  }

GATED_ADD_OP(float, f32)
GATED_ADD_OP(__half, f16)
GATED_ADD_OP(__nv_bfloat16, bf16)

// Appends a shared-expert slot to every token's routing: ids gain `shared_id`, weights gain
// sigmoid(gate[t]). ids/weights are [tokens, top_k]; the outputs [tokens, top_k + 1].
template <typename T>
__global__ void moe_append_shared_slot_kernel(const unsigned int *__restrict__ ids,
                                              const float *__restrict__ weights,
                                              const T *__restrict__ gate,
                                              unsigned int *__restrict__ ids_out,
                                              float *__restrict__ weights_out, const int tokens,
                                              const int top_k, const unsigned int shared_id) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = top_k + 1;
  if (i >= tokens * width) {
    return;
  }
  const int t = i / width, j = i - t * width;
  if (j < top_k) {
    ids_out[i] = ids[t * top_k + j];
    weights_out[i] = weights[t * top_k + j];
  } else {
    ids_out[i] = shared_id;
    weights_out[i] = 1.0f / (1.0f + expf(-to_float(gate[t])));
  }
}

#define APPEND_SHARED_OP(T, SUFFIX)                                                                \
  extern "C" void moe_append_shared_slot_##SUFFIX(const unsigned int *ids, const float *weights,   \
                                                  const void *gate, unsigned int *ids_out,          \
                                                  float *weights_out, int tokens, int top_k,        \
                                                  unsigned int shared_id, int64_t stream) {         \
    const int threads = 128;                                                                        \
    const int n = tokens * (top_k + 1);                                                             \
    moe_append_shared_slot_kernel<T><<<(n + threads - 1) / threads, threads, 0,                     \
                                       (cudaStream_t)stream>>>(ids, weights, (const T *)gate,       \
                                                               ids_out, weights_out, tokens, top_k, \
                                                               shared_id);                          \
  }

APPEND_SHARED_OP(float, f32)
APPEND_SHARED_OP(__half, f16)
APPEND_SHARED_OP(__nv_bfloat16, bf16)

// Router and shared-expert gate logits in one launch: `w` stacks the `n_router` router rows
// and the single shared-gate row ([n_router + 1, k]); one wave per (token, row), f32 dots.
// router_out: [tokens, n_router]; shared_out: [tokens].
template <typename T>
__global__ void moe_dual_gate_gemv_kernel(const T *__restrict__ x, const T *__restrict__ w,
                                          T *__restrict__ router_out, T *__restrict__ shared_out,
                                          const int tokens, const int n_router, const int k) {
  const int lane = threadIdx.x % 32;
  const int item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int rows = n_router + 1;
  if (item >= tokens * rows) {
    return;
  }
  const int t = item / rows, r = item - t * rows;
  const T *xr = x + (int64_t)t * k;
  const T *wr = w + (int64_t)r * k;
  float acc = 0.0f;
  // 8 elements (16 bytes) per lane per step; k is a multiple of 256.
  for (int c = lane * 8; c < k; c += 32 * 8) {
    const uint4 xv = *reinterpret_cast<const uint4 *>(xr + c);
    const uint4 wv = *reinterpret_cast<const uint4 *>(wr + c);
    const T *xe = reinterpret_cast<const T *>(&xv);
    const T *we = reinterpret_cast<const T *>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc = fmaf(to_float(xe[j]), to_float(we[j]), acc);
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_xor_sync(0xffffffff, acc, off, 32);
  }
  if (lane == 0) {
    if (r < n_router) {
      router_out[(int64_t)t * n_router + r] = from_float<T>(acc);
    } else {
      shared_out[t] = from_float<T>(acc);
    }
  }
}

#define DUAL_GATE_OP(T, SUFFIX)                                                                    \
  extern "C" void moe_dual_gate_gemv_##SUFFIX(const void *x, const void *w, void *router_out,      \
                                              void *shared_out, int tokens, int n_router, int k,   \
                                              int64_t stream) {                                     \
    const int warps = 8;                                                                            \
    const int items = tokens * (n_router + 1);                                                      \
    moe_dual_gate_gemv_kernel<T><<<(items + warps - 1) / warps, warps * 32, 0,                     \
                                   (cudaStream_t)stream>>>((const T *)x, (const T *)w,             \
                                                           (T *)router_out, (T *)shared_out,        \
                                                           tokens, n_router, k);                    \
  }

DUAL_GATE_OP(float, f32)
DUAL_GATE_OP(__half, f16)
DUAL_GATE_OP(__nv_bfloat16, bf16)

// out = act(x * scale), contiguous with n elements.
template <typename T>
__global__ void act_scaled_kernel(const T *__restrict__ x, T *__restrict__ out, const int64_t n,
                                  const float scale, const int kind) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = from_float<T>(glu_activation(to_float(x[i]) * scale, kind));
  }
}

#define ACT_SCALED_OP(T, SUFFIX)                                                                  \
  extern "C" void act_scaled_##SUFFIX(const void *x, void *out, int64_t n, float scale, int kind, \
                                      int64_t stream) {                                           \
    const int threads = 256;                                                                      \
    const int64_t blocks = (n + threads - 1) / threads;                                           \
    act_scaled_kernel<T><<<(unsigned)(blocks < 65535 ? blocks : 65535), threads, 0,               \
                           (cudaStream_t)stream>>>((const T *)x, (T *)out, n, scale, kind);       \
  }

ACT_SCALED_OP(float, f32)
ACT_SCALED_OP(__half, f16)
ACT_SCALED_OP(__nv_bfloat16, bf16)

// QSA keys plus the indexer's aux head: out[b, h, t, :] = k[b, h, t, :] for h < heads, and
// out[b, heads, t, :] = [raw_key[b, t, :d_idx], cos[b, t, :half], sin[b, t, :half], 0...].
// k and out are contiguous; raw_key rows are `raw_stride` apart; cos/sin rows are
// contiguous with `cs_batch_stride` (0 when broadcast) between batches.
template <typename T>
__global__ void qsa_pack_keys_kernel(const T *__restrict__ k, const T *__restrict__ raw_key,
                                     const T *__restrict__ cos, const T *__restrict__ sin,
                                     T *__restrict__ out, const int heads, const int seq,
                                     const int d, const int d_idx, const int half,
                                     const int64_t raw_stride, const int64_t cs_batch_stride,
                                     const int64_t n) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int j = (int)(i % d);
    const int64_t row = i / d;
    const int t = (int)(row % seq);
    const int h = (int)((row / seq) % (heads + 1));
    const int64_t b = row / seq / (heads + 1);
    T value;
    if (h < heads) {
      value = k[((b * heads + h) * seq + t) * d + j];
    } else if (j < d_idx) {
      value = raw_key[(b * seq + t) * raw_stride + j];
    } else if (j < d_idx + half) {
      value = cos[b * cs_batch_stride + (int64_t)t * half + (j - d_idx)];
    } else if (j < d_idx + 2 * half) {
      value = sin[b * cs_batch_stride + (int64_t)t * half + (j - d_idx - half)];
    } else {
      value = from_float<T>(0.0f);
    }
    out[i] = value;
  }
}

#define QSA_PACK_KEYS_OP(T, SUFFIX)                                                                \
  extern "C" void qsa_pack_keys_##SUFFIX(const void *k, const void *raw_key, const void *cos,      \
                                         const void *sin, void *out, int batch, int heads, int seq, \
                                         int d, int d_idx, int half, int64_t raw_stride,           \
                                         int64_t cs_batch_stride, int64_t stream) {                \
    const int64_t n = (int64_t)batch * (heads + 1) * seq * d;                                      \
    const int threads = 256;                                                                       \
    const int64_t blocks = (n + threads - 1) / threads;                                            \
    qsa_pack_keys_kernel<T><<<(unsigned)(blocks < 65535 ? blocks : 65535), threads, 0,             \
                              (cudaStream_t)stream>>>((const T *)k, (const T *)raw_key,            \
                                                      (const T *)cos, (const T *)sin, (T *)out,    \
                                                      heads, seq, d, d_idx, half, raw_stride,      \
                                                      cs_batch_stride, n);                         \
  }

QSA_PACK_KEYS_OP(float, f32)
QSA_PACK_KEYS_OP(__half, f16)
QSA_PACK_KEYS_OP(__nv_bfloat16, bf16)

// dst[rows[r], :] = src[r, :] for contiguous rows of `row_elems` 2- or 4-byte elements.
template <typename W>
__global__ void indexed_row_copy_kernel(const W *__restrict__ src, W *__restrict__ dst,
                                        const uint32_t *__restrict__ rows, const int64_t n_rows,
                                        const int64_t row_elems) {
  const int64_t n = n_rows * row_elems;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int64_t r = i / row_elems;
    dst[(int64_t)rows[r] * row_elems + (i - r * row_elems)] = src[i];
  }
}

extern "C" void indexed_row_copy(const void *src, void *dst, const uint32_t *rows, int64_t n_rows,
                                 int64_t row_elems, int elem_bytes, int64_t stream) {
  const int64_t n = n_rows * row_elems;
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  const unsigned grid = (unsigned)(blocks < 65535 ? blocks : 65535);
  if (elem_bytes == 4) {
    indexed_row_copy_kernel<uint32_t><<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const uint32_t *)src, (uint32_t *)dst, rows, n_rows, row_elems);
  } else {
    indexed_row_copy_kernel<uint16_t><<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const uint16_t *)src, (uint16_t *)dst, rows, n_rows, row_elems);
  }
}
