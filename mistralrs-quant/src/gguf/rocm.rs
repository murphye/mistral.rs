//! ROCm implementation of indexed MoE forward for GGUF quantized weights.
//!
//! Uses candle's indexed MoE kernels, which read only the selected experts,
//! instead of dequantizing every expert as the generic path does.

use candle_core::{
    quantized::{GgmlDType, QMatMul},
    DType, Result, Tensor,
};

/// Weight types the candle indexed MoE kernel supports.
fn supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
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
    // The kernel takes f32/f16/bf16 activations and returns their dtype.
    let x3 = match x3.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => x3.contiguous()?,
        _ => x3.to_dtype(DType::F32)?,
    };
    let ids2 = ids2.to_dtype(DType::U32)?.contiguous()?;
    let out = qtensor.indexed_moe_forward(&x3, &ids2)?;
    let out = match restore {
        Some((b, s)) => {
            let (_, topk, n) = out.dims3()?;
            out.reshape((b, s, topk, n))?
        }
        None => out,
    };
    Ok(Some(out.to_dtype(x.dtype())?))
}
