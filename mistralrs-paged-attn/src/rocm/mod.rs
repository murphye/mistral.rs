//! ROCm backend: the CUDA paged attention kernels compiled with hipcc
//! (`USE_ROCM`), launched through candle's ROCm device.
//!
//! This covers the standard paged cache layout. The FlashInfer, FA3, MLA and
//! attention sink kernels are CUDA only. fp8 (OCP e4m3) KV caches use the shared
//! conversion paths, compiled with ENABLE_FP8.

// The HIP build compiles the fp8 (OCP e4m3) KV cache paths with ENABLE_FP8 (see build.rs).
pub const USE_FP8: bool = true;

mod backend;
mod ffi;

pub use backend::{
    copy_blocks, gather_kv_cache, kv_scale_update, paged_attention, paged_attention_with_window,
    reshape_and_cache, swap_blocks,
};
