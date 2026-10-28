//! Fused GLU activation, MoE combines and partial NeoX RoPE.

use std::ffi::c_void;

use candle_core::{DType, Result, Tensor};

use crate::rocm::rocm_ptr;

fn half_or_f32_ptr(t: &Tensor, name: &str) -> Result<u64> {
    match t.dtype() {
        DType::F32 => rocm_ptr::<f32>(t, name),
        DType::F16 => rocm_ptr::<half::f16>(t, name),
        DType::BF16 => rocm_ptr::<half::bf16>(t, name),
        dt => candle_core::bail!("{name}: unsupported dtype {dt:?}"),
    }
}

fn stream(t: &Tensor) -> Result<i64> {
    Ok(t.device().as_rocm_device()?.hip_stream().raw() as i64)
}

/// `act(gate) * up` in one kernel; `kind` is a `mistralrs_quant::GluActivationType`.
pub fn glu_act_mul(gate: &Tensor, up: &Tensor, kind: i32) -> Result<Tensor> {
    if gate.shape() != up.shape() || gate.dtype() != up.dtype() {
        candle_core::bail!("glu_act_mul: gate and up must have the same shape and dtype");
    }
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let out = unsafe { Tensor::empty(gate.shape(), gate.dtype(), gate.device()) }?;
    let launch = match gate.dtype() {
        DType::F32 => crate::rocm::ffi::glu_act_mul_f32,
        DType::F16 => crate::rocm::ffi::glu_act_mul_f16,
        DType::BF16 => crate::rocm::ffi::glu_act_mul_bf16,
        dt => candle_core::bail!("glu_act_mul: unsupported dtype {dt:?}"),
    };
    unsafe {
        launch(
            half_or_f32_ptr(&gate, "gate")? as *const c_void,
            half_or_f32_ptr(&up, "up")? as *const c_void,
            half_or_f32_ptr(&out, "out")? as *mut c_void,
            gate.elem_count() as i64,
            kind,
            stream(&gate)?,
        );
    }
    Ok(out)
}

/// `sum_k ys[t, k, :] * weights[t, k]` for `ys: [tokens, top_k, hidden]` and f32
/// `weights: [tokens, top_k]`, accumulated in f32 and returned in `ys`' dtype.
pub fn moe_weighted_reduce(ys: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (tokens, top_k, hidden) = ys.dims3()?;
    if weights.dtype() != DType::F32 || weights.dims() != [tokens, top_k] {
        candle_core::bail!("moe_weighted_reduce: weights must be f32 [{tokens}, {top_k}]");
    }
    let ys = ys.contiguous()?;
    let weights = weights.contiguous()?;
    let out = unsafe { Tensor::empty((tokens, hidden), ys.dtype(), ys.device()) }?;
    let launch = match ys.dtype() {
        DType::F32 => crate::rocm::ffi::moe_weighted_reduce_f32,
        DType::F16 => crate::rocm::ffi::moe_weighted_reduce_f16,
        DType::BF16 => crate::rocm::ffi::moe_weighted_reduce_bf16,
        dt => candle_core::bail!("moe_weighted_reduce: unsupported dtype {dt:?}"),
    };
    unsafe {
        launch(
            half_or_f32_ptr(&ys, "ys")? as *const c_void,
            rocm_ptr::<f32>(&weights, "weights")? as *const f32,
            half_or_f32_ptr(&out, "out")? as *mut c_void,
            tokens as i32,
            top_k as i32,
            hidden as i32,
            stream(&ys)?,
        );
    }
    Ok(out)
}

/// NeoX RoPE over the first `2 * half` dims of `x: [b, s, heads, d]`, computed in
/// f32, with `cos`/`sin` of shape `[b, s, half]`.
pub fn rope_partial_neox(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, s, heads, d) = x.dims4()?;
    let half = cos.dim(candle_core::D::Minus1)?;
    if cos.elem_count() != b * s * half || sin.elem_count() != b * s * half || 2 * half > d {
        candle_core::bail!("rope_partial_neox: bad cos/sin shapes {:?} for x {:?}", cos.shape(), x.shape());
    }
    let x = x.contiguous()?;
    let cos = cos.to_dtype(DType::F32)?.contiguous()?;
    let sin = sin.to_dtype(DType::F32)?.contiguous()?;
    let out = unsafe { Tensor::empty(x.shape(), x.dtype(), x.device()) }?;
    let launch = match x.dtype() {
        DType::F32 => crate::rocm::ffi::rope_partial_neox_f32,
        DType::F16 => crate::rocm::ffi::rope_partial_neox_f16,
        DType::BF16 => crate::rocm::ffi::rope_partial_neox_bf16,
        dt => candle_core::bail!("rope_partial_neox: unsupported dtype {dt:?}"),
    };
    unsafe {
        launch(
            half_or_f32_ptr(&x, "x")? as *const c_void,
            rocm_ptr::<f32>(&cos, "cos")? as *const f32,
            rocm_ptr::<f32>(&sin, "sin")? as *const f32,
            half_or_f32_ptr(&out, "out")? as *mut c_void,
            x.elem_count() as i64,
            heads as i32,
            d as i32,
            half as i32,
            stream(&x)?,
        );
    }
    Ok(out)
}

/// `y + sigmoid(gate) * shared` for `y`, `shared` of shape `[.., hidden]` and one
/// gate logit per row, all in the same dtype.
pub fn add_sigmoid_gated(y: &Tensor, shared: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let hidden = y.dim(candle_core::D::Minus1)?;
    if y.shape() != shared.shape()
        || gate.elem_count() * hidden != y.elem_count()
        || y.dtype() != shared.dtype()
        || y.dtype() != gate.dtype()
    {
        candle_core::bail!("add_sigmoid_gated: mismatched inputs {:?} {:?} {:?}", y.shape(), shared.shape(), gate.shape());
    }
    let y = y.contiguous()?;
    let shared = shared.contiguous()?;
    let gate = gate.contiguous()?;
    let out = unsafe { Tensor::empty(y.shape(), y.dtype(), y.device()) }?;
    let launch = match y.dtype() {
        DType::F32 => crate::rocm::ffi::add_sigmoid_gated_f32,
        DType::F16 => crate::rocm::ffi::add_sigmoid_gated_f16,
        DType::BF16 => crate::rocm::ffi::add_sigmoid_gated_bf16,
        dt => candle_core::bail!("add_sigmoid_gated: unsupported dtype {dt:?}"),
    };
    unsafe {
        launch(
            half_or_f32_ptr(&y, "y")? as *const c_void,
            half_or_f32_ptr(&shared, "shared")? as *const c_void,
            half_or_f32_ptr(&gate, "gate")? as *const c_void,
            half_or_f32_ptr(&out, "out")? as *mut c_void,
            y.elem_count() as i64,
            hidden as i32,
            stream(&y)?,
        );
    }
    Ok(out)
}

/// `act(x * scale)` in one kernel; `kind` is a `mistralrs_quant::GluActivationType`.
pub fn act_scaled(x: &Tensor, scale: f32, kind: i32) -> Result<Tensor> {
    let x = x.contiguous()?;
    let out = unsafe { Tensor::empty(x.shape(), x.dtype(), x.device()) }?;
    let launch = match x.dtype() {
        DType::F32 => crate::rocm::ffi::act_scaled_f32,
        DType::F16 => crate::rocm::ffi::act_scaled_f16,
        DType::BF16 => crate::rocm::ffi::act_scaled_bf16,
        dt => candle_core::bail!("act_scaled: unsupported dtype {dt:?}"),
    };
    unsafe {
        launch(
            half_or_f32_ptr(&x, "x")? as *const c_void,
            half_or_f32_ptr(&out, "out")? as *mut c_void,
            x.elem_count() as i64,
            scale,
            kind,
            stream(&x)?,
        );
    }
    Ok(out)
}

/// QSA keys `k: [b, heads, s, d]` with the indexer's aux head appended as head `heads`:
/// `[raw_key, cos, sin]` zero-padded to `d`. `raw_key: [b, s, d_idx]` may have strided rows;
/// `cos`/`sin: [b or 1, s, half]`.
pub fn qsa_pack_keys(k: &Tensor, raw_key: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, heads, seq, d) = k.dims4()?;
    let (_, _, d_idx) = raw_key.dims3()?;
    let half = cos.dim(candle_core::D::Minus1)?;
    if d_idx + 2 * half > d {
        candle_core::bail!("qsa_pack_keys: aux width {} exceeds head dim {d}", d_idx + 2 * half);
    }
    let k = k.contiguous()?;
    let dtype = k.dtype();
    let raw_key = if raw_key.dtype() == dtype { raw_key.clone() } else { raw_key.to_dtype(dtype)? };
    // Rows of raw_key must be evenly spaced with unit stride inside a row.
    let raw_key = match raw_key.stride() {
        [sb, ss, 1] if *sb == seq * *ss => raw_key,
        _ => raw_key.contiguous()?,
    };
    let raw_stride = raw_key.stride()[1] as i64;
    let cos = cos.to_dtype(dtype)?.contiguous()?;
    let sin = sin.to_dtype(dtype)?.contiguous()?;
    if cos.elem_count() != sin.elem_count() || cos.elem_count() % (seq * half) != 0 {
        candle_core::bail!("qsa_pack_keys: cos/sin shapes {:?} {:?}", cos.shape(), sin.shape());
    }
    let cs_batch_stride = if cos.elem_count() == seq * half { 0 } else { (seq * half) as i64 };
    let out = unsafe { Tensor::empty((b, heads + 1, seq, d), dtype, k.device()) }?;
    let launch = match dtype {
        DType::F32 => crate::rocm::ffi::qsa_pack_keys_f32,
        DType::F16 => crate::rocm::ffi::qsa_pack_keys_f16,
        DType::BF16 => crate::rocm::ffi::qsa_pack_keys_bf16,
        dt => candle_core::bail!("qsa_pack_keys: unsupported dtype {dt:?}"),
    };
    unsafe {
        launch(
            half_or_f32_ptr(&k, "k")? as *const c_void,
            half_or_f32_ptr(&raw_key, "raw_key")? as *const c_void,
            half_or_f32_ptr(&cos, "cos")? as *const c_void,
            half_or_f32_ptr(&sin, "sin")? as *const c_void,
            half_or_f32_ptr(&out, "out")? as *mut c_void,
            b as i32,
            heads as i32,
            seq as i32,
            d as i32,
            d_idx as i32,
            half as i32,
            raw_stride,
            cs_batch_stride,
            stream(&k)?,
        );
    }
    Ok(out)
}
