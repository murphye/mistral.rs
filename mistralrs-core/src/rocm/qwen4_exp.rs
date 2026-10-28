//! ROCm launchers for the Qwen4-Exp hyper-connection kernels in
//! `src/cuda/qwen4_exp.cu`, compiled with hipcc. Mirrors the CUDA wrappers.
#![allow(clippy::cast_possible_truncation)]

use candle_core::{DType, Device, Result, Storage, Tensor};
use core::ffi::c_void;

mod ffi {
    use core::ffi::c_void;

    extern "C" {
        pub(super) fn qwen4_hc_norm(
            x: *const c_void,
            weight: *const f32,
            out: *mut c_void,
            rows: i32,
            hc: i32,
            hidden: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_hc_mix(
            xn: *const c_void,
            gate: *const c_void,
            out: *mut c_void,
            tokens: i32,
            hc: i32,
            hidden: i32,
            dtype: i32,
            stream: i64,
        );
        #[allow(clippy::too_many_arguments)]
        pub(super) fn qwen4_hc_combine(
            res: *const c_void,
            block_out: *const c_void,
            inject: *const c_void,
            res_out: *mut c_void,
            next_weight: *const f32,
            xn_out: *mut c_void,
            rows: i32,
            hc: i32,
            hidden: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
    }
}

/// Device address of a tensor's first element. The tensor must stay alive while it is used.
fn dev_ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Rocm(storage) = &*storage else {
        candle_core::bail!("expected a ROCm tensor");
    };
    macro_rules! ptr {
        ($ty:ty) => {{
            storage
                .as_rocm_slice::<$ty>()?
                .slice(layout.start_offset()..)
                .device_ptr()
        }};
    }
    Ok(match t.dtype() {
        DType::BF16 => ptr!(half::bf16),
        DType::F16 => ptr!(half::f16),
        DType::F32 => ptr!(f32),
        other => candle_core::bail!("unsupported dtype {other:?} for a raw device pointer"),
    })
}

fn opt_ptr(t: Option<&Tensor>) -> Result<u64> {
    t.map_or(Ok(0), dev_ptr)
}

fn stream(device: &Device) -> Result<i64> {
    Ok(device.as_rocm_device()?.hip_stream().raw() as i64)
}

fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("Qwen4-Exp kernels do not support {other:?}"),
    }
}

fn empty(shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    // Safety: every element is written by the kernel before it is read.
    unsafe { Tensor::empty(shape, dtype, device) }
}

/// Grouped RMSNorm over each of the `hc` streams of `x: [tokens, hc * hidden]`.
pub(crate) fn hc_norm(x: &Tensor, weight: &Tensor, hc: usize, eps: f64) -> Result<Tensor> {
    let x = x.contiguous()?;
    let width = x.dim(candle_core::D::Minus1)?;
    let rows = x.elem_count() / (width / hc);
    let out = empty(x.dims(), x.dtype(), x.device())?;
    unsafe {
        ffi::qwen4_hc_norm(
            dev_ptr(&x)? as *const c_void,
            dev_ptr(weight)? as *const f32,
            dev_ptr(&out)? as *mut c_void,
            rows as i32,
            hc as i32,
            (width / hc) as i32,
            eps as f32,
            dtype_code(x.dtype())?,
            stream(x.device())?,
        );
    }
    Ok(out)
}

/// Mean over streams of `xn * sigmoid(gate)`: `[.., hc * hidden] -> [.., hidden]`.
pub(crate) fn hc_mix(xn: &Tensor, gate: &Tensor, hc: usize) -> Result<Tensor> {
    let xn = xn.contiguous()?;
    let gate = gate.contiguous()?;
    let width = xn.dim(candle_core::D::Minus1)?;
    let hidden = width / hc;
    let tokens = xn.elem_count() / width;
    let mut shape = xn.dims().to_vec();
    *shape.last_mut().expect("rank >= 1") = hidden;
    let out = empty(&shape, xn.dtype(), xn.device())?;
    unsafe {
        ffi::qwen4_hc_mix(
            dev_ptr(&xn)? as *const c_void,
            dev_ptr(&gate)? as *const c_void,
            dev_ptr(&out)? as *mut c_void,
            tokens as i32,
            hc as i32,
            hidden as i32,
            dtype_code(xn.dtype())?,
            stream(xn.device())?,
        );
    }
    Ok(out)
}

/// `res + block_out * 2 * sigmoid(inject / hc)` per stream, plus the next mixer's grouped norm.
pub(crate) fn hc_combine(
    res: &Tensor,
    block_out: &Tensor,
    inject: &Tensor,
    next_norm: Option<&Tensor>,
    hc: usize,
    eps: f64,
) -> Result<(Tensor, Option<Tensor>)> {
    let res = res.contiguous()?;
    let block_out = block_out.contiguous()?;
    let inject = inject.contiguous()?;
    let width = res.dim(candle_core::D::Minus1)?;
    let hidden = width / hc;
    let rows = res.elem_count() / hidden;
    let out = empty(res.dims(), res.dtype(), res.device())?;
    let xn = next_norm
        .map(|_| empty(res.dims(), res.dtype(), res.device()))
        .transpose()?;
    unsafe {
        ffi::qwen4_hc_combine(
            dev_ptr(&res)? as *const c_void,
            dev_ptr(&block_out)? as *const c_void,
            dev_ptr(&inject)? as *const c_void,
            dev_ptr(&out)? as *mut c_void,
            opt_ptr(next_norm)? as *const f32,
            opt_ptr(xn.as_ref())? as *mut c_void,
            rows as i32,
            hc as i32,
            hidden as i32,
            eps as f32,
            dtype_code(res.dtype())?,
            stream(res.device())?,
        );
    }
    Ok((out, xn))
}
