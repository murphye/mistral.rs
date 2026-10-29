//! Fused MoE router (score, top-k and weight normalization in one kernel).

use std::ffi::c_void;

use candle_core::{DType, Result, Tensor};

use crate::ops::{MoeRouterTopKConfig, TopKOutput};
use crate::rocm::rocm_ptr;

/// ROCm counterpart of `cuda_moe_router_topk`: f32 weights and u32 expert ids,
/// both shaped like `logits` with the last dimension replaced by `top_k`.
pub fn moe_router_topk_rocm(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    let mut out_dims = logits.dims().to_vec();
    *out_dims.last_mut().unwrap() = config.top_k;
    let dev = logits.device();
    let values = unsafe { Tensor::empty(out_dims.as_slice(), DType::F32, dev) }?;
    let indices = unsafe { Tensor::empty(out_dims.as_slice(), DType::U32, dev) }?;
    moe_router_topk_rocm_into(logits, config, selection_bias, expert_scale, &values, &indices)?;
    Ok(TopKOutput { values, indices })
}

/// [`moe_router_topk_rocm`] writing into `values` (f32) and `indices` (u32) shaped like
/// `logits` with the last dimension replaced by a stride of at least `top_k`: the first
/// `top_k` columns of each row are written, the rest left for the caller (a shared-expert
/// slot appended without another launch).
pub fn moe_router_topk_rocm_into(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
    values: &Tensor,
    indices: &Tensor,
) -> Result<()> {
    let logits = logits.contiguous()?;
    let n_experts = *logits
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".to_string()))?;
    if config.top_k == 0 || config.top_k > n_experts {
        candle_core::bail!(
            "moe_router_topk_rocm top_k={} must be in [1, {n_experts}]",
            config.top_k
        );
    }
    let f32_param = |t: Option<&Tensor>, name: &str| -> Result<(Option<Tensor>, *const c_void)> {
        let Some(t) = t else {
            return Ok((None, std::ptr::null()));
        };
        if t.elem_count() != n_experts {
            candle_core::bail!("moe_router_topk_rocm {name} must have {n_experts} elements");
        }
        let t = t.to_dtype(DType::F32)?.contiguous()?;
        let ptr = rocm_ptr::<f32>(&t, name)? as *const c_void;
        Ok((Some(t), ptr))
    };
    // The tensors are kept alive until the kernel has been queued.
    let (_bias, bias_ptr) = f32_param(selection_bias, "selection_bias")?;
    let (_scale, scale_ptr) = f32_param(expert_scale, "expert_scale")?;

    let nrows = logits.elem_count() / n_experts;
    let out_stride = *values.dims().last().unwrap_or(&0);
    if out_stride < config.top_k
        || values.dims() != indices.dims()
        || values.elem_count() != nrows * out_stride
        || values.dtype() != DType::F32
        || indices.dtype() != DType::U32
        || !values.is_contiguous()
        || !indices.is_contiguous()
    {
        candle_core::bail!(
            "moe_router_topk_rocm_into: outputs {:?}/{:?} don't fit {nrows} rows of top_k {}",
            values.shape(),
            indices.shape(),
            config.top_k
        );
    }
    let dev = logits.device();
    let (clamp_min, clamp_max, clamp_logits) = match config.logit_clip {
        Some((min, max)) => (min, max, true),
        None => (0.0, 0.0, false),
    };
    let values_ptr = rocm_ptr::<f32>(values, "values")? as *mut c_void;
    let indices_ptr = rocm_ptr::<u32>(indices, "indices")? as *mut c_void;
    let stream = dev.as_rocm_device()?.hip_stream().raw() as i64;
    macro_rules! launch {
        ($ty:ty, $ffi_fn:ident) => {
            unsafe {
                crate::rocm::ffi::$ffi_fn(
                    rocm_ptr::<$ty>(&logits, "logits")? as *const c_void,
                    values_ptr,
                    indices_ptr,
                    bias_ptr,
                    scale_ptr,
                    nrows as i32,
                    n_experts as i32,
                    config.top_k as i32,
                    out_stride as i32,
                    config.score_function.as_i32(),
                    config.selected_weight.as_i32(),
                    config.renormalize,
                    clamp_logits,
                    clamp_min,
                    clamp_max,
                    config.norm_min,
                    config.output_scale,
                    stream,
                )
            }
        };
    }
    match logits.dtype() {
        DType::BF16 => launch!(half::bf16, moe_router_topk_bf16),
        DType::F16 => launch!(half::f16, moe_router_topk_f16),
        DType::F32 => launch!(f32, moe_router_topk_f32),
        dt => candle_core::bail!("moe_router_topk_rocm unsupported dtype: {dt:?}"),
    }
    Ok(())
}
