#pragma once
// ROCm builds define USE_ROCM, which selects this header. Reuse the NVIDIA
// implementation: candle_hip_prelude.h maps the CUDA fp8 names it uses onto HIP,
// and the build passes ENABLE_FP8 so the fp8 KV cache paths compile (HIP's
// hip_fp8.h provides the OCP e4m3 conversions, hardware-backed on gfx12 and
// gfx942/950).
// Include the dtype headers in ROCm mode first, so the NVIDIA header below
// reuses them instead of pulling in their CUDA inline assembly.
#include "attention/attention_dtypes.h"

namespace vllm {
// HIP's __half2_raw holds __half_raw members (CUDA's holds unsigned shorts), so the
// NVIDIA fp8 conversions' half_to_float(res.x) needs this overload.
inline __device__ float half_to_float(__half_raw h) { return half_to_float(h.x); }
} // namespace vllm

#pragma push_macro("USE_ROCM")
#undef USE_ROCM
#include "quantization/fp8/nvidia/quant_utils.cuh"
#pragma pop_macro("USE_ROCM")
