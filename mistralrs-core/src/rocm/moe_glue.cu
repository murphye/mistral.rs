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
