#pragma once
// ROCm builds define USE_ROCM, which selects this header. Reuse the NVIDIA
// implementation: candle_hip_prelude.h maps the CUDA fp8 names it uses onto HIP.
// Without ENABLE_FP8 the fp8 conversions are compiled out, so fp8 KV caches are
// not available on ROCm yet.
// Include the dtype headers in ROCm mode first, so the NVIDIA header below
// reuses them instead of pulling in their CUDA inline assembly.
#include "attention/attention_dtypes.h"
#pragma push_macro("USE_ROCM")
#undef USE_ROCM
#include "quantization/fp8/nvidia/quant_utils.cuh"
#pragma pop_macro("USE_ROCM")
