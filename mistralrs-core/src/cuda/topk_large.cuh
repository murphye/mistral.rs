// Large-vocabulary top-k for token sampling, shared by the CUDA and ROCm builds.
#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

template <typename T>
__device__ __forceinline__ T warp_reduce_max_with_idx(T val, int idx,
                                                      int &max_idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T other_val = __shfl_down_sync(0xffffffff, val, offset);
    int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val ||
        (other_val == val && other_idx >= 0 && (idx < 0 || other_idx < idx))) {
      val = other_val;
      idx = other_idx;
    }
  }
  max_idx = idx;
  return val;
}

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float block_reduce_sum_f32(float val) {
  __shared__ float warp_sums[32];
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (blockDim.x + 31) / 32;

  val = warp_reduce_sum_f32(val);
  if (lane_id == 0) {
    warp_sums[warp_id] = val;
  }
  __syncthreads();

  val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
  if (warp_id == 0) {
    val = warp_reduce_sum_f32(val);
  }
  return val;
}

// Large-vocabulary top-k for token sampling. The MoE top-k kernel above stages
// a full row in shared memory, which is not viable for 100k+ vocabularies. This
// kernel scans fixed-size chunks, emits per-chunk top-k candidates, and can
// compute each chunk's contribution to the full softmax denominator.
template <typename T> __device__ __forceinline__ float topk_to_float(T x) {
  return static_cast<float>(x);
}

template <> __device__ __forceinline__ float topk_to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float
topk_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T, bool BATCHED, bool COMPUTE_SOFTMAX>
__global__ void topk_large_stage1(
    const T *__restrict__ input, float *__restrict__ block_values,
    uint32_t *__restrict__ block_indices, float *__restrict__ block_maxes,
    float *__restrict__ block_sums, const int ncols, const int k,
    const int chunk_size, const float *__restrict__ inv_temperatures,
    const float scalar_inv_temperature) {
  const size_t row = BATCHED ? blockIdx.y : 0;
  const int chunk = blockIdx.x;
  if constexpr (BATCHED) {
    const size_t candidate_stride = static_cast<size_t>(gridDim.x) * k;
    const size_t block_stride = gridDim.x;
    input += row * static_cast<size_t>(ncols);
    block_values += row * candidate_stride;
    block_indices += row * candidate_stride;
    if constexpr (COMPUTE_SOFTMAX) {
      block_maxes += row * block_stride;
      block_sums += row * block_stride;
    }
  }
  const int start = chunk * chunk_size;
  const int end = min(start + chunk_size, ncols);
  const int width = max(0, end - start);
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  extern __shared__ char smem[];
  bool *s_used = reinterpret_cast<bool *>(smem);

  for (int i = tid; i < chunk_size; i += block_size) {
    s_used[i] = false;
  }
  __syncthreads();

  for (int ki = 0; ki < k; ++ki) {
    float local_max = -INFINITY;
    int local_idx = -1;

    for (int local = tid; local < width; local += block_size) {
      const float candidate = topk_to_float(input[start + local]);
      if (!s_used[local] && candidate == candidate && candidate > local_max) {
        local_max = candidate;
        local_idx = start + local;
      }
    }

    int warp_max_idx;
    float warp_max =
        warp_reduce_max_with_idx<float>(local_max, local_idx, warp_max_idx);

    __shared__ float warp_maxes[32];
    __shared__ int warp_indices[32];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (block_size + 31) / 32;

    if (lane_id == 0) {
      warp_maxes[warp_id] = warp_max;
      warp_indices[warp_id] = warp_max_idx;
    }
    __syncthreads();

    if (tid < 32) {
      float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
      int idx = (tid < num_warps) ? warp_indices[tid] : -1;
      int final_idx;
      float final_max = warp_reduce_max_with_idx<float>(val, idx, final_idx);

      if (tid == 0) {
        block_values[chunk * k + ki] = final_max;
        block_indices[chunk * k + ki] =
            final_idx >= 0 ? static_cast<uint32_t>(final_idx) : 0;
        if (final_idx >= start && final_idx < end) {
          s_used[final_idx - start] = true;
        }
      }
    }
    __syncthreads();
  }

  if constexpr (COMPUTE_SOFTMAX) {
    const float inv_temperature =
        BATCHED ? inv_temperatures[row] : scalar_inv_temperature;
    const float block_max =
        width > 0 ? block_values[chunk * k] * inv_temperature : -INFINITY;
    float local_sum = 0.0f;
    for (int local = tid; local < width; local += block_size) {
      const float candidate = topk_to_float(input[start + local]);
      if (candidate != candidate) {
        local_sum = NAN;
      } else if (block_max != -INFINITY) {
        local_sum += expf(candidate * inv_temperature - block_max);
      }
    }

    const float block_sum = block_reduce_sum_f32(local_sum);
    if (tid == 0) {
      block_maxes[chunk] = block_max;
      block_sums[chunk] = block_sum;
    }
  }
}

template <bool BATCHED, bool COMPUTE_SOFTMAX>
__global__ void topk_large_stage2_f32_packed(
    const float *__restrict__ block_values,
    const uint32_t *__restrict__ block_indices,
    const float *__restrict__ block_maxes, const float *__restrict__ block_sums,
    float *__restrict__ packed_out, const int nblocks, const int k) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int n_candidates = nblocks * k;
  if constexpr (BATCHED) {
    const size_t row = blockIdx.x;
    const size_t candidate_offset = row * static_cast<size_t>(n_candidates);
    block_values += candidate_offset;
    block_indices += candidate_offset;
    if constexpr (COMPUTE_SOFTMAX) {
      const size_t block_offset = row * static_cast<size_t>(nblocks);
      block_maxes += block_offset;
      block_sums += block_offset;
    }
    packed_out +=
        row * static_cast<size_t>(2 * k + (COMPUTE_SOFTMAX ? 2 : 0));
  }

  extern __shared__ char smem[];
  bool *s_used = reinterpret_cast<bool *>(smem);

  for (int i = tid; i < n_candidates; i += block_size) {
    s_used[i] = false;
  }
  __syncthreads();

  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (block_size + 31) / 32;

  if constexpr (COMPUTE_SOFTMAX) {
    float local_global_max = -INFINITY;
    for (int block = tid; block < nblocks; block += block_size) {
      local_global_max = fmaxf(local_global_max, block_maxes[block]);
    }

    int unused_idx;
    float warp_global_max =
        warp_reduce_max_with_idx<float>(local_global_max, tid, unused_idx);

    __shared__ float warp_maxes[32];
    if (lane_id == 0) {
      warp_maxes[warp_id] = warp_global_max;
    }
    __syncthreads();

    __shared__ float s_global_max;
    if (tid < 32) {
      float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
      int final_idx;
      float final_max = warp_reduce_max_with_idx<float>(val, tid, final_idx);
      if (tid == 0) {
        s_global_max = final_max;
      }
    }
    __syncthreads();

    float local_denom = 0.0f;
    if (s_global_max != -INFINITY) {
      for (int block = tid; block < nblocks; block += block_size) {
        local_denom +=
            block_sums[block] * expf(block_maxes[block] - s_global_max);
      }
    }
    const float denom = block_reduce_sum_f32(local_denom);
    if (tid == 0) {
      packed_out[2 * k] = denom;
      packed_out[2 * k + 1] = s_global_max;
    }
    __syncthreads();
  }

  for (int ki = 0; ki < k; ++ki) {
    float local_max = -INFINITY;
    int local_pos = -1;

    for (int pos = tid; pos < n_candidates; pos += block_size) {
      const float candidate = block_values[pos];
      if (!s_used[pos] && candidate == candidate && candidate > local_max) {
        local_max = candidate;
        local_pos = pos;
      }
    }

    int warp_max_pos;
    float warp_max =
        warp_reduce_max_with_idx<float>(local_max, local_pos, warp_max_pos);

    __shared__ float merge_warp_maxes[32];
    __shared__ int merge_warp_indices[32];

    if (lane_id == 0) {
      merge_warp_maxes[warp_id] = warp_max;
      merge_warp_indices[warp_id] = warp_max_pos;
    }
    __syncthreads();

    if (tid < 32) {
      float val = (tid < num_warps) ? merge_warp_maxes[tid] : -INFINITY;
      int idx = (tid < num_warps) ? merge_warp_indices[tid] : -1;
      int final_pos;
      float final_max = warp_reduce_max_with_idx<float>(val, idx, final_pos);

      if (tid == 0) {
        packed_out[ki] = final_max;
        packed_out[k + ki] = final_pos >= 0
                                 ? static_cast<float>(block_indices[final_pos])
                                 : 0.0f;
        if (final_pos >= 0) {
          s_used[final_pos] = true;
        }
      }
    }
    __syncthreads();
  }
}
