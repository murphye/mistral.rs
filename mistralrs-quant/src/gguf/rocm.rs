//! ROCm implementation of indexed MoE forward for GGUF quantized weights.
//!
//! Uses candle's fused q8_1 indexed MoE kernel, which reads only the selected
//! experts, instead of dequantizing every expert as the generic path does.

use candle_core::{
    quantized::{GgmlDType, QMatMul},
    DType, Result, Tensor,
};

/// Weight types the candle indexed MoE kernel supports.
fn supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

/// Runs `x @ w[ids]^T` for weights `[num_experts, n, k]`, input
/// `[tokens, topk_or_1, k]` and indices `[tokens, topk]`, returning
/// `[tokens, topk, n]` in the input dtype. The 4D/5D input layouts used by some
/// models are flattened to that form. Returns `None` when unsupported.
pub fn rocm_indexed_moe_forward(
    qmatmul: &QMatMul,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Option<Tensor>> {
    let QMatMul::QTensor(qtensor) = qmatmul else {
        return Ok(None);
    };
    if !supported(qtensor.dtype()) || qtensor.shape().rank() != 3 {
        return Ok(None);
    }
    let (x3, ids2, restore): (Tensor, Tensor, Option<(usize, usize)>) = match *x.dims() {
        [_, _, _] if ids.rank() == 2 => (x.clone(), ids.clone(), None),
        [b, s, xt, h] if ids.rank() == 3 => {
            let (ib, is, t) = ids.dims3()?;
            if ib != b || is != s {
                return Ok(None);
            }
            (
                x.reshape((b * s, xt, h))?,
                ids.reshape((b * s, t))?,
                Some((b, s)),
            )
        }
        [b, s, 1, 1, h] if ids.rank() == 3 => {
            let (ib, is, t) = ids.dims3()?;
            if ib != b || is != s {
                return Ok(None);
            }
            (
                x.reshape((b * s, 1, h))?,
                ids.reshape((b * s, t))?,
                Some((b, s)),
            )
        }
        _ => return Ok(None),
    };
    let original_dtype = x.dtype();
    // The kernel reads f32 activations and u32 ids from the start of their buffers.
    let x3 = if x3.dtype() == DType::F32 {
        x3.copy()?
    } else {
        x3.to_dtype(DType::F32)?
    };
    let ids2 = if ids2.dtype() == DType::U32 {
        ids2.copy()?
    } else {
        ids2.to_dtype(DType::U32)?
    };
    let out = qtensor.indexed_moe_forward(&x3, &ids2)?;
    let out = match restore {
        Some((b, s)) => {
            let (_, topk, n) = out.dims3()?;
            out.reshape((b, s, topk, n))?
        }
        None => out,
    };
    Ok(Some(out.to_dtype(original_dtype)?))
}
