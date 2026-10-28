//! ROCm backend: the CUDA paged attention kernels compiled with hipcc
//! (`USE_ROCM`), launched through candle's ROCm device.
//!
//! This covers the standard paged cache layout. The FlashInfer, FA3, MLA and
//! attention sink kernels are CUDA only, and fp8 KV caches are not supported yet.

/// FP8 KV caches need the fp8 conversion kernels, which are not built for ROCm.
pub const USE_FP8: bool = false;

mod backend;
mod ffi;

pub use backend::{
    copy_blocks, gather_kv_cache, kv_scale_update, paged_attention, paged_attention_with_window,
    reshape_and_cache, swap_blocks,
};
