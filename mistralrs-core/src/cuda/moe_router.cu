// Fused MoE router: score function, top-k selection and weight normalization
// in one warp per row. Shared by the CUDA and ROCm builds.
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

constexpr int MOE_ROUTER_SCORE_RAW = 0;
constexpr int MOE_ROUTER_SCORE_SOFTMAX = 1;
constexpr int MOE_ROUTER_SCORE_SIGMOID = 2;
constexpr int MOE_ROUTER_WEIGHT_SCORE = 0;
constexpr int MOE_ROUTER_WEIGHT_SOFTMAX = 1;
constexpr int MOE_ROUTER_WEIGHT_SIGMOID = 2;
constexpr int MOE_ROUTER_ROWS_PER_BLOCK = 4;

template <typename T> __device__ __forceinline__ float router_to_float(T x) {
  return static_cast<float>(x);
}

template <> __device__ __forceinline__ float router_to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float
router_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float router_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset, 32);
  }
  return val;
}

__device__ __forceinline__ float router_warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
  }
  return val;
}

template <int values_per_thread, bool use_limit>
__device__ __forceinline__ void
router_softmax_warp(float (&vals)[values_per_thread], const int limit,
                    const int lane) {
  float max_val = -INFINITY;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      max_val = max(max_val, vals[i]);
    }
  }

  max_val = router_warp_reduce_max(max_val);
  float sum = 0.0f;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      vals[i] = expf(vals[i] - max_val);
      sum += vals[i];
    } else {
      vals[i] = 0.0f;
    }
  }

  sum = router_warp_reduce_sum(sum);
  const float inv_sum = 1.0f / sum;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      vals[i] *= inv_sum;
    }
  }
}

template <int values_per_thread, bool use_limit>
__device__ __forceinline__ void
router_sigmoid_warp(float (&vals)[values_per_thread], const int limit,
                    const int lane) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    vals[i] = (!use_limit || idx < limit) ? 1.0f / (1.0f + expf(-vals[i]))
                                          : -INFINITY;
  }
}

template <typename T, int n_experts, bool has_bias, bool has_expert_scale>
__launch_bounds__(MOE_ROUTER_ROWS_PER_BLOCK * 32, 1) __global__
    void moe_router_topk_kernel(
        const T *__restrict__ logits, float *__restrict__ weights,
        uint32_t *__restrict__ ids, const float *__restrict__ selection_bias,
        const float *__restrict__ expert_scale, const int n_rows,
        const int top_k, const int score_mode, const int weight_mode,
        const bool renormalize, const bool clamp_logits, const float clamp_min,
        const float clamp_max, const float norm_min, const float output_scale) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= n_rows) {
    return;
  }

  logits += row * n_experts;
  weights += row * top_k;
  ids += row * top_k;

  constexpr int values_per_thread = n_experts > 32 ? n_experts / 32 : 1;

  float raw[values_per_thread];
  float score[values_per_thread];
  float selection[values_per_thread];
  float output_weights[values_per_thread];
  uint32_t output_ids[values_per_thread];

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int expert = lane + i * 32;
    float value = (n_experts % 32 == 0 || expert < n_experts)
                      ? router_to_float(logits[expert])
                      : -INFINITY;
    if (clamp_logits && expert < n_experts) {
      value = fminf(fmaxf(value, clamp_min), clamp_max);
    }
    if (value != value) {
      value = -INFINITY;
    }
    raw[i] = value;
    score[i] = value;
    selection[i] = value;
    output_weights[i] = 0.0f;
    output_ids[i] = 0;
  }

  if (score_mode == MOE_ROUTER_SCORE_SOFTMAX) {
    router_softmax_warp<values_per_thread, false>(score, n_experts, lane);
  } else if (score_mode == MOE_ROUTER_SCORE_SIGMOID) {
    router_sigmoid_warp<values_per_thread, false>(score, n_experts, lane);
  }

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int expert = lane + i * 32;
    selection[i] = score[i];
    if constexpr (has_bias) {
      if (expert < n_experts) {
        selection[i] += selection_bias[expert];
      }
    }
    if (selection[i] != selection[i]) {
      selection[i] = -INFINITY;
    }
  }

  for (int k_idx = 0; k_idx < top_k; k_idx++) {
    float best_selection = selection[0];
    float best_score = score[0];
    float best_raw = raw[0];
    int best_expert = lane;

#pragma unroll
    for (int i = 1; i < values_per_thread; i++) {
      const int expert = lane + i * 32;
      if ((n_experts % 32 == 0 || expert < n_experts) &&
          selection[i] > best_selection) {
        best_selection = selection[i];
        best_score = score[i];
        best_raw = raw[i];
        best_expert = expert;
      }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      const float other_selection =
          __shfl_xor_sync(0xffffffff, best_selection, mask, 32);
      const float other_score =
          __shfl_xor_sync(0xffffffff, best_score, mask, 32);
      const float other_raw = __shfl_xor_sync(0xffffffff, best_raw, mask, 32);
      const int other_expert =
          __shfl_xor_sync(0xffffffff, best_expert, mask, 32);
      if (other_selection > best_selection ||
          (other_selection == best_selection && other_expert < best_expert)) {
        best_selection = other_selection;
        best_score = other_score;
        best_raw = other_raw;
        best_expert = other_expert;
      }
    }

    float out = best_score;
    if (weight_mode == MOE_ROUTER_WEIGHT_SOFTMAX) {
      out = best_raw;
    } else if (weight_mode == MOE_ROUTER_WEIGHT_SIGMOID) {
      out = 1.0f / (1.0f + expf(-best_raw));
    }

    if ((k_idx & 31) == lane) {
      output_weights[k_idx / 32] = out;
      output_ids[k_idx / 32] = static_cast<uint32_t>(best_expert);
    }

    if ((best_expert & 31) == lane) {
      selection[best_expert / 32] = -INFINITY;
    }
  }

  if (weight_mode == MOE_ROUTER_WEIGHT_SOFTMAX) {
    router_softmax_warp<values_per_thread, true>(output_weights, top_k, lane);
  }

  if (renormalize) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      const int idx = lane + i * 32;
      if (idx < top_k) {
        sum += output_weights[i];
      }
    }
    sum = router_warp_reduce_sum(sum);
    sum = fmaxf(sum, norm_min);
    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      output_weights[i] *= inv_sum;
    }
  }

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (idx < top_k) {
      float scale = output_scale;
      if constexpr (has_expert_scale) {
        scale *= expert_scale[output_ids[i]];
      }
      weights[idx] = output_weights[i] * scale;
      ids[idx] = output_ids[i];
    }
  }
}

template <typename T, bool has_bias, bool has_expert_scale>
void launch_moe_router_topk(const T *logits, float *weights, uint32_t *ids,
                            const float *selection_bias,
                            const float *expert_scale, const int n_rows,
                            const int n_experts, const int top_k,
                            const int score_mode, const int weight_mode,
                            const bool renormalize, const bool clamp_logits,
                            const float clamp_min, const float clamp_max,
                            const float norm_min, const float output_scale,
                            int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block_dims(32, MOE_ROUTER_ROWS_PER_BLOCK, 1);
  dim3 grid_dims((n_rows + MOE_ROUTER_ROWS_PER_BLOCK - 1) /
                     MOE_ROUTER_ROWS_PER_BLOCK,
                 1, 1);

#define LAUNCH_MOE_ROUTER_TOPK(EXPERTS)                                        \
  moe_router_topk_kernel<T, EXPERTS, has_bias, has_expert_scale>               \
      <<<grid_dims, block_dims, 0, custream>>>(                                \
          logits, weights, ids, selection_bias, expert_scale, n_rows, top_k,   \
          score_mode, weight_mode, renormalize, clamp_logits, clamp_min,       \
          clamp_max, norm_min, output_scale);                                  \
  break

  switch (n_experts) {
  case 1:
    LAUNCH_MOE_ROUTER_TOPK(1);
  case 2:
    LAUNCH_MOE_ROUTER_TOPK(2);
  case 4:
    LAUNCH_MOE_ROUTER_TOPK(4);
  case 8:
    LAUNCH_MOE_ROUTER_TOPK(8);
  case 16:
    LAUNCH_MOE_ROUTER_TOPK(16);
  case 32:
    LAUNCH_MOE_ROUTER_TOPK(32);
  case 64:
    LAUNCH_MOE_ROUTER_TOPK(64);
  case 128:
    LAUNCH_MOE_ROUTER_TOPK(128);
  case 256:
    LAUNCH_MOE_ROUTER_TOPK(256);
  case 512:
    LAUNCH_MOE_ROUTER_TOPK(512);
  case 288:
    LAUNCH_MOE_ROUTER_TOPK(288);
  case 576:
    LAUNCH_MOE_ROUTER_TOPK(576);
  default:
    break;
  }

#undef LAUNCH_MOE_ROUTER_TOPK
}

template <typename T>
void moe_router_topk_dispatch(const T *logits, float *weights, uint32_t *ids,
                              const float *selection_bias,
                              const float *expert_scale, const int n_rows,
                              const int n_experts, const int top_k,
                              const int score_mode, const int weight_mode,
                              const bool renormalize, const bool clamp_logits,
                              const float clamp_min, const float clamp_max,
                              const float norm_min, const float output_scale,
                              int64_t stream) {
  if (selection_bias != nullptr && expert_scale != nullptr) {
    launch_moe_router_topk<T, true, true>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else if (selection_bias != nullptr) {
    launch_moe_router_topk<T, true, false>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else if (expert_scale != nullptr) {
    launch_moe_router_topk<T, false, true>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else {
    launch_moe_router_topk<T, false, false>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  }
}

extern "C" void moe_router_topk_f32(const float *logits, float *weights,
                                    uint32_t *ids, const float *selection_bias,
                                    const float *expert_scale, int n_rows,
                                    int n_experts, int top_k, int score_mode,
                                    int weight_mode, bool renormalize,
                                    bool clamp_logits, float clamp_min,
                                    float clamp_max, float norm_min,
                                    float output_scale, int64_t stream) {
  moe_router_topk_dispatch<float>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}

extern "C" void
moe_router_topk_bf16(const __nv_bfloat16 *logits, float *weights, uint32_t *ids,
                     const float *selection_bias, const float *expert_scale,
                     int n_rows, int n_experts, int top_k, int score_mode,
                     int weight_mode, bool renormalize, bool clamp_logits,
                     float clamp_min, float clamp_max, float norm_min,
                     float output_scale, int64_t stream) {
  moe_router_topk_dispatch<__nv_bfloat16>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}

extern "C" void moe_router_topk_f16(const __half *logits, float *weights,
                                    uint32_t *ids, const float *selection_bias,
                                    const float *expert_scale, int n_rows,
                                    int n_experts, int top_k, int score_mode,
                                    int weight_mode, bool renormalize,
                                    bool clamp_logits, float clamp_min,
                                    float clamp_max, float norm_min,
                                    float output_scale, int64_t stream) {
  moe_router_topk_dispatch<__half>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}
