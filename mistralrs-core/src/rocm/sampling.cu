// Large-vocabulary top-k for token sampling: per-chunk candidates plus each
// chunk's softmax partial sums, merged into `[values(k), indices(k), denom, max]`.
#include "topk_large.cuh"

#define TOPK_PACKED(T, SUFFIX)                                                                    \
  extern "C" void topk_large_packed_##SUFFIX(                                                     \
      const void *input, float *block_values, uint32_t *block_indices, float *block_maxes,        \
      float *block_sums, float *packed_out, int ncols, int k, int chunk_size, int nblocks,        \
      float inv_temperature, int64_t stream) {                                                    \
    constexpr int block_size = 256;                                                               \
    const size_t stage1_smem = static_cast<size_t>(chunk_size) * sizeof(bool);                   \
    const size_t stage2_smem = static_cast<size_t>(nblocks) * static_cast<size_t>(k) * sizeof(bool); \
    topk_large_stage1<T, false, true><<<nblocks, block_size, stage1_smem, (cudaStream_t)stream>>>( \
        (const T *)input, block_values, block_indices, block_maxes, block_sums, ncols, k,         \
        chunk_size, nullptr, inv_temperature);                                                    \
    topk_large_stage2_f32_packed<false, true><<<1, block_size, stage2_smem, (cudaStream_t)stream>>>( \
        block_values, block_indices, block_maxes, block_sums, packed_out, nblocks, k);            \
  }

TOPK_PACKED(float, f32)
TOPK_PACKED(__half, f16)
TOPK_PACKED(__nv_bfloat16, bf16)
