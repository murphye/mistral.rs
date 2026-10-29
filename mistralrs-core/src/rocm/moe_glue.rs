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
/// `ids: [tokens, top_k]` u32 and `weights: [tokens, top_k]` f32 with one more slot per token:
/// expert `shared_id` weighted by `sigmoid(gate[t])` (`gate: [tokens]` or `[tokens, 1]`).
pub fn moe_append_shared_slot(
    ids: &Tensor,
    weights: &Tensor,
    gate: &Tensor,
    shared_id: u32,
) -> Result<(Tensor, Tensor)> {
    let (tokens, top_k) = ids.dims2()?;
    if weights.dims() != [tokens, top_k]
        || gate.elem_count() != tokens
        || ids.dtype() != DType::U32
        || weights.dtype() != DType::F32
    {
        candle_core::bail!(
            "moe_append_shared_slot: mismatched inputs {:?} {:?} {:?}",
            ids.shape(),
            weights.shape(),
            gate.shape()
        );
    }
    let ids = ids.contiguous()?;
    let weights = weights.contiguous()?;
    let gate = gate.contiguous()?;
    let ids_out = unsafe { Tensor::empty((tokens, top_k + 1), DType::U32, ids.device()) }?;
    let weights_out = unsafe { Tensor::empty((tokens, top_k + 1), DType::F32, ids.device()) }?;
    let launch = match gate.dtype() {
        DType::F32 => crate::rocm::ffi::moe_append_shared_slot_f32,
        DType::F16 => crate::rocm::ffi::moe_append_shared_slot_f16,
        DType::BF16 => crate::rocm::ffi::moe_append_shared_slot_bf16,
        dt => candle_core::bail!("moe_append_shared_slot: unsupported gate dtype {dt:?}"),
    };
    unsafe {
        launch(
            rocm_ptr::<u32>(&ids, "ids")? as *const u32,
            rocm_ptr::<f32>(&weights, "weights")? as *const f32,
            half_or_f32_ptr(&gate, "gate")? as *const c_void,
            rocm_ptr::<u32>(&ids_out, "ids_out")? as *mut u32,
            rocm_ptr::<f32>(&weights_out, "weights_out")? as *mut f32,
            tokens as i32,
            top_k as i32,
            shared_id,
            stream(&ids)?,
        );
    }
    Ok((ids_out, weights_out))
}

/// Router logits `[tokens, n_router]` and the shared-expert gate logit `[tokens]` from one
/// launch over `w: [n_router + 1, k]` (the router rows, then the shared gate row).
pub fn moe_dual_gate_gemv(x: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    moe_dual_gate_gemv_slots(x, w, None)
}

/// [`moe_dual_gate_gemv`] that also fills the shared-expert slot of a routing table: with
/// `slots = (weights [tokens, stride] f32, ids [tokens, stride] u32, shared_id)` the last column
/// of each row receives `sigmoid(shared logit)` and `shared_id`.
pub fn moe_dual_gate_gemv_slots(
    x: &Tensor,
    w: &Tensor,
    slots: Option<(&Tensor, &Tensor, u32)>,
) -> Result<(Tensor, Tensor)> {
    let (tokens, k) = x.dims2()?;
    let (rows, w_k) = w.dims2()?;
    if w_k != k || rows < 2 || k % 256 != 0 || x.dtype() != w.dtype() {
        candle_core::bail!("moe_dual_gate_gemv: mismatched inputs {:?} {:?}", x.shape(), w.shape());
    }
    let n_router = rows - 1;
    let x = x.contiguous()?;
    let w = w.contiguous()?;
    let router = unsafe { Tensor::empty((tokens, n_router), x.dtype(), x.device()) }?;
    let shared = unsafe { Tensor::empty(tokens, x.dtype(), x.device()) }?;
    let launch = match x.dtype() {
        DType::F32 => crate::rocm::ffi::moe_dual_gate_gemv_f32,
        DType::F16 => crate::rocm::ffi::moe_dual_gate_gemv_f16,
        DType::BF16 => crate::rocm::ffi::moe_dual_gate_gemv_bf16,
        dt => candle_core::bail!("moe_dual_gate_gemv: unsupported dtype {dt:?}"),
    };
    let (slot_weights, slot_ids, slot_stride, shared_id) = match slots {
        Some((weights, ids, shared_id)) => {
            let (rows, stride) = weights.dims2()?;
            if rows != tokens
                || ids.dims() != weights.dims()
                || weights.dtype() != DType::F32
                || ids.dtype() != DType::U32
                || !weights.is_contiguous()
                || !ids.is_contiguous()
            {
                candle_core::bail!(
                    "moe_dual_gate_gemv_slots: slot tables {:?}/{:?} don't fit {tokens} tokens",
                    weights.shape(),
                    ids.shape()
                );
            }
            (
                rocm_ptr::<f32>(weights, "slot_weights")? as *mut f32,
                rocm_ptr::<u32>(ids, "slot_ids")? as *mut u32,
                stride as i32,
                shared_id,
            )
        }
        None => (std::ptr::null_mut(), std::ptr::null_mut(), 0, 0),
    };
    unsafe {
        launch(
            half_or_f32_ptr(&x, "x")? as *const c_void,
            half_or_f32_ptr(&w, "w")? as *const c_void,
            half_or_f32_ptr(&router, "router")? as *mut c_void,
            half_or_f32_ptr(&shared, "shared")? as *mut c_void,
            slot_weights,
            slot_ids,
            slot_stride,
            shared_id,
            tokens as i32,
            n_router as i32,
            k as i32,
            stream(&x)?,
        );
    }
    Ok((router, shared))
}

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

/// `dst[rows[i]] = src[i]` along the first dimension, in place on `dst` (bf16, f16 or f32
/// state pools indexed by u32 slot rows).
pub fn copy_rows(src: &Tensor, dst: &Tensor, rows: &Tensor) -> Result<()> {
    let src = src.contiguous()?;
    let n_rows = src.dim(0)?;
    if src.dtype() != dst.dtype()
        || rows.dtype() != DType::U32
        || rows.dim(0)? != n_rows
        || src.dims()[1..] != dst.dims()[1..]
        || !dst.is_contiguous()
    {
        candle_core::bail!(
            "copy_rows: src {:?} {:?}, dst {:?} {:?}, rows {:?} {:?}",
            src.shape(),
            src.dtype(),
            dst.shape(),
            dst.dtype(),
            rows.shape(),
            rows.dtype()
        );
    }
    let row_elems = src.elem_count() / n_rows.max(1);
    if n_rows == 0 || row_elems == 0 {
        return Ok(());
    }
    let rows = rows.contiguous()?;
    unsafe {
        crate::rocm::ffi::indexed_row_copy(
            half_or_f32_ptr(&src, "src")? as *const c_void,
            half_or_f32_ptr(dst, "dst")? as *mut c_void,
            rocm_ptr::<u32>(&rows, "rows")? as *const u32,
            n_rows as i64,
            row_elems as i64,
            src.dtype().size_in_bytes() as i32,
            stream(&src)?,
        );
    }
    Ok(())
}
