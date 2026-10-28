//! ROCm builds of CUDA kernels shared with `src/cuda`.
pub mod ffi;
pub mod gdn;
pub mod moe_glue;
pub mod moe_router;
pub mod qwen4_exp;
pub mod sampling;

use candle_core::{Result, Tensor};

/// Device address of a ROCm tensor's first element.
pub(crate) fn rocm_ptr<T: candle_core::rocm_backend::RocmDType>(
    t: &Tensor,
    name: &str,
) -> Result<u64> {
    let (s, l) = t.storage_and_layout();
    let s = match &*s {
        candle_core::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
        _ => candle_core::bail!("{name} must be a rocm tensor"),
    };
    Ok(s.slice(l.start_offset()..).device_ptr())
}
