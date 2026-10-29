//! ROCm launchers for the gated delta net kernels shared with the CUDA backend
//! (`src/cuda/gdn.cu`, compiled with hipcc). Covers the gating, causal
//! convolution, gated RMSNorm and the key-major recurrence used by the generic
//! GDN path; the CUDA-only fused, value-major and speculative paths are not
//! ported.
use candle_core::{DType, Result, Tensor};

use crate::cuda::gdn::{FusedDecodeRecurrence, GdnStateSlots, RecurrenceInputs};
use crate::kv_cache::RecurrentStateLayout;
use crate::rocm::rocm_ptr;
use crate::gdn::GdnGateActivation;

// Must match the state dtype codes in gdn.cu.
const GDN_STATE_DTYPE_F16: i32 = 0;
const GDN_STATE_DTYPE_BF16: i32 = 1;
const GDN_STATE_DTYPE_F32: i32 = 2;

/// Recurrence kernels available on ROCm.
#[derive(Clone, Copy, Debug)]
pub enum RecurrenceKernel {
    /// One token at a time; used for decode and short prefills.
    Scalar,
    /// One warp per value column.
    Warp,
    /// 64-token chunks.
    Chunked,
}

/// Gated delta rule recurrence. `state` is the key-major `[BH, K, V]` state (or
/// the pooled layout with `Pooled` slots), updated in place. Returns `[BH, S, V]` f32.
pub fn gated_delta_rule_recurrence_rocm(
    kernel: RecurrenceKernel,
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    use candle_core as candle;

    let RecurrenceInputs { q, k, v, g, beta } = inputs;
    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;
    let dev = q.device().as_rocm_device()?;

    fn f32_ptr(t: &Tensor, name: &str) -> Result<*mut f32> {
        let (s, l) = t.storage_and_layout();
        let s = match &*s {
            candle_core::Storage::Rocm(c) => c.as_rocm_slice::<f32>()?,
            _ => candle_core::bail!("{name} must be a rocm tensor"),
        };
        Ok(s.slice(l.start_offset()..).device_ptr() as *mut f32)
    }
    let q_ptr = f32_ptr(q, "q")?;
    let k_ptr = f32_ptr(k, "k")?;
    let v_ptr = f32_ptr(v, "v")?;
    let g_ptr = f32_ptr(g, "g")?;
    let beta_ptr = f32_ptr(beta, "beta")?;
    let (state_ptr, state_dtype) = rocm_recurrent_state_ptr(state, "state")?;

    let output_buf = unsafe { dev.alloc::<f32>(bh * seq_len * v_dim) }?;
    let stream = dev.hip_stream().raw() as i64;
    let launcher = match kernel {
        RecurrenceKernel::Scalar => crate::rocm::ffi::gated_delta_rule_recurrence,
        RecurrenceKernel::Warp => crate::rocm::ffi::warp_gated_delta_rule_recurrence,
        RecurrenceKernel::Chunked => crate::rocm::ffi::chunked_gated_delta_rule_recurrence,
    };
    with_slot_indices(slots, |slot_ptr, batch| {
        let num_heads = bh.checked_div(batch).unwrap_or(1);
        unsafe {
            launcher(
                q_ptr,
                k_ptr,
                v_ptr,
                g_ptr,
                beta_ptr,
                state_ptr,
                output_buf.device_ptr() as *mut f32,
                bh as i32,
                seq_len as i32,
                k_dim as i32,
                v_dim as i32,
                slot_ptr,
                num_heads as i32,
                state_dtype,
                stream,
            );
        }
        Ok(())
    })?;

    let output_storage = candle::RocmStorage::wrap_rocm_slice(output_buf, dev.clone());
    Ok(Tensor::from((
        candle::Storage::Rocm(output_storage),
        (bh, seq_len, v_dim),
    )))
}

fn rocm_recurrent_state_ptr(tensor: &Tensor, name: &str) -> Result<(*mut core::ffi::c_void, i32)> {
    use candle_core as candle;

    let (storage, layout) = tensor.storage_and_layout();
    let offset = layout.start_offset();
    let (pointer, dtype) = match tensor.dtype() {
        DType::F16 => {
            let storage = match &*storage {
                candle::Storage::Rocm(storage) => storage.as_rocm_slice::<half::f16>()?,
                _ => candle::bail!("{name} must be a ROCm tensor"),
            };
            (storage.slice(offset..).device_ptr(), GDN_STATE_DTYPE_F16)
        }
        DType::BF16 => {
            let storage = match &*storage {
                candle::Storage::Rocm(storage) => storage.as_rocm_slice::<half::bf16>()?,
                _ => candle::bail!("{name} must be a ROCm tensor"),
            };
            (storage.slice(offset..).device_ptr(), GDN_STATE_DTYPE_BF16)
        }
        DType::F32 => {
            let storage = match &*storage {
                candle::Storage::Rocm(storage) => storage.as_rocm_slice::<f32>()?,
                _ => candle::bail!("{name} must be a ROCm tensor"),
            };
            (storage.slice(offset..).device_ptr(), GDN_STATE_DTYPE_F32)
        }
        dtype => candle::bail!("{name} has unsupported recurrent state dtype {dtype:?}"),
    };
    Ok((pointer as *mut core::ffi::c_void, dtype))
}

fn with_slot_indices<T>(
    slots: GdnStateSlots<'_>,
    f: impl FnOnce(*const i32, usize) -> Result<T>,
) -> Result<T> {
    match slots {
        GdnStateSlots::Gathered => f(std::ptr::null(), 0),
        GdnStateSlots::Pooled(slots) => {
            let batch = slots.dim(0)?;
            let (s, l) = slots.storage_and_layout();
            let s = match &*s {
                candle_core::Storage::Rocm(c) => c.as_rocm_slice::<u32>()?,
                _ => candle_core::bail!("slot indices must be a rocm tensor"),
            };
            let ptr = s.slice(l.start_offset()..).device_ptr() as *const i32;
            f(ptr, batch)
        }
    }
}

pub fn causal_conv1d_rocm(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
    is_update: bool,
    slots: GdnStateSlots<'_>,
) -> Result<(Tensor, Tensor)> {
    use candle_core as candle;
    use core::ffi::c_void;

    fn rocm_fwd<T: candle::rocm_backend::RocmDType + candle::rocm_backend::hip::DeviceRepr>(
        x: &Tensor,
        weight: &Tensor,
        conv_state: &Tensor,
        kernel_size: usize,
        is_update: bool,
        slots: GdnStateSlots<'_>,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor)> {
        let dev = x.device().as_rocm_device()?;
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let pooled = matches!(slots, GdnStateSlots::Pooled(_));

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
            _ => candle::bail!("x must be a rocm tensor"),
        };
        let x_offset = x_l.start_offset();
        let x_stride = x_l.stride();

        let (w_s, w_l) = weight.storage_and_layout();
        let w_s = match &*w_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
            _ => candle::bail!("weight must be a rocm tensor"),
        };
        let w_offset = w_l.start_offset();

        let stream = dev.hip_stream().raw() as i64;

        if is_update {
            // Clone conv_state so the kernel can mutate it in place
            let conv_state_new = conv_state.clone();

            let output_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim) }?;

            // Scope the borrow of conv_state_new so we can move it later
            {
                let (cs_s, cs_l) = conv_state_new.storage_and_layout();
                let cs_s = match &*cs_s {
                    candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
                    _ => candle::bail!("conv_state must be a rocm tensor"),
                };
                let cs_offset = cs_l.start_offset();

                with_slot_indices(slots, |slot_ptr, _| {
                    unsafe {
                        crate::rocm::ffi::causal_conv1d_update(
                            x_s.slice(x_offset..).device_ptr() as *const c_void,
                            w_s.slice(w_offset..).device_ptr() as *const c_void,
                            cs_s.slice(cs_offset..).device_ptr() as *mut c_void,
                            output_buf.device_ptr() as *mut c_void,
                            batch_size as i32,
                            conv_dim as i32,
                            kernel_size as i32,
                            x_stride[0] as i64,
                            x_stride[1] as i64,
                            x_stride[2] as i64,
                            slot_ptr,
                            dtype_code,
                            stream,
                        );
                    }
                    Ok(())
                })?;
            }

            let output_storage = candle::RocmStorage::wrap_rocm_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Rocm(output_storage),
                (batch_size, 1usize, conv_dim),
            ));

            Ok((output, conv_state_new))
        } else {
            let output_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim * seq_len) }?;
            // Pooled: the save kernel rewrites the pool rows in place (it reads ahead of every write)
            let cs_buf = if pooled {
                None
            } else {
                Some(unsafe { dev.alloc::<T>(batch_size * conv_dim * kernel_size) }?)
            };
            let (cs_s, cs_l) = conv_state.storage_and_layout();
            let cs_s = match &*cs_s {
                candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
                _ => candle::bail!("conv_state must be a rocm tensor"),
            };
            let cs_offset = cs_l.start_offset();
            let cs_in_ptr = cs_s.slice(cs_offset..).device_ptr();
            let cs_out_ptr = match &cs_buf {
                Some(buf) => buf.device_ptr(),
                None => cs_in_ptr,
            };

            with_slot_indices(slots, |slot_ptr, _| {
                unsafe {
                    crate::rocm::ffi::causal_conv1d_full(
                        x_s.slice(x_offset..).device_ptr() as *const c_void,
                        w_s.slice(w_offset..).device_ptr() as *const c_void,
                        cs_in_ptr as *const c_void,
                        cs_out_ptr as *mut c_void,
                        output_buf.device_ptr() as *mut c_void,
                        batch_size as i32,
                        conv_dim as i32,
                        seq_len as i32,
                        kernel_size as i32,
                        x_stride[0] as i64,
                        x_stride[1] as i64,
                        x_stride[2] as i64,
                        slot_ptr,
                        dtype_code,
                        stream,
                    );
                }
                Ok(())
            })?;

            let output_storage = candle::RocmStorage::wrap_rocm_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Rocm(output_storage),
                (batch_size, seq_len, conv_dim),
            ));

            let new_conv_state = match cs_buf {
                Some(cs_buf) => Tensor::from((
                    candle::Storage::Rocm(candle::RocmStorage::wrap_rocm_slice(
                        cs_buf,
                        dev.clone(),
                    )),
                    (batch_size, conv_dim, kernel_size),
                )),
                None => conv_state.clone(),
            };

            Ok((output, new_conv_state))
        }
    }

    let weight = weight.contiguous()?;
    if matches!(slots, GdnStateSlots::Pooled(_)) && !conv_state.is_contiguous() {
        candle_core::bail!("pooled conv state must be contiguous");
    }
    let conv_state = conv_state.contiguous()?;
    match x.dtype() {
        DType::F16 => {
            rocm_fwd::<half::f16>(x, &weight, &conv_state, kernel_size, is_update, slots, 0)
        }
        DType::BF16 => {
            rocm_fwd::<half::bf16>(x, &weight, &conv_state, kernel_size, is_update, slots, 1)
        }
        other => candle_core::bail!("causal_conv1d_rocm only supports f16/bf16, got {:?}", other),
    }
}

fn normalize_gdn_rmsnorm_layout(
    dims: &[usize],
    strides: &[usize],
    hidden_dim: usize,
) -> Result<([usize; 4], [usize; 4])> {
    match (dims, strides) {
        ([d0, d1], [s0, s1]) if d1 % hidden_dim == 0 => Ok((
            [1, *d0, d1 / hidden_dim, hidden_dim],
            [0, *s0, hidden_dim * *s1, *s1],
        )),
        ([d0, d1, d2], [s0, s1, s2]) if d2 % hidden_dim == 0 => Ok((
            [*d0, *d1, d2 / hidden_dim, hidden_dim],
            [*s0, *s1, hidden_dim * *s2, *s2],
        )),
        ([d0, d1, d2, d3], [s0, s1, s2, s3]) if *d3 == hidden_dim => {
            Ok(([*d0, *d1, *d2, *d3], [*s0, *s1, *s2, *s3]))
        }
        _ => candle_core::bail!(
            "gated RMSNorm expects rank 2-4 with a final dimension divisible by {hidden_dim}"
        ),
    }
}

pub fn rmsnorm_gated_rocm(
    x: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
    activation: GdnGateActivation,
) -> Result<Tensor> {
    use candle_core as candle;
    use core::ffi::c_void;

    fn rocm_fwd<T: candle::rocm_backend::RocmDType + candle::rocm_backend::hip::DeviceRepr>(
        x: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f64,
        activation: GdnGateActivation,
        dtype_code: i32,
    ) -> Result<Tensor> {
        let weight = weight.contiguous()?;
        let hidden_dim = weight.dims1()?;
        let dev = x.device().as_rocm_device()?;

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
            _ => candle::bail!("x must be a rocm tensor"),
        };
        let x_offset = x_l.start_offset();
        let (dims, x_stride) = normalize_gdn_rmsnorm_layout(x.dims(), x_l.stride(), hidden_dim)?;

        let (gate_s, gate_l) = gate.storage_and_layout();
        let gate_s = match &*gate_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
            _ => candle::bail!("gate must be a rocm tensor"),
        };
        let gate_offset = gate_l.start_offset();
        let (gate_dims, gate_stride) =
            normalize_gdn_rmsnorm_layout(gate.dims(), gate_l.stride(), hidden_dim)?;
        if gate_dims != dims {
            candle::bail!("gated RMSNorm inputs have incompatible logical shapes");
        }

        let (weight_s, weight_l) = weight.storage_and_layout();
        let weight_s = match &*weight_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
            _ => candle::bail!("weight must be a rocm tensor"),
        };
        let weight_offset = weight_l.start_offset();

        let rows = dims[0] * dims[1] * dims[2];
        let output_buf = unsafe { dev.alloc::<T>(rows * hidden_dim) }?;
        let stream = dev.hip_stream().raw() as i64;

        unsafe {
            crate::rocm::ffi::gdn_rmsnorm_gated(
                x_s.slice(x_offset..).device_ptr() as *const c_void,
                gate_s.slice(gate_offset..).device_ptr() as *const c_void,
                weight_s.slice(weight_offset..).device_ptr() as *const c_void,
                output_buf.device_ptr() as *mut c_void,
                rows as i32,
                hidden_dim as i32,
                dims[1] as i32,
                dims[2] as i32,
                x_stride[0] as i64,
                x_stride[1] as i64,
                x_stride[2] as i64,
                x_stride[3] as i64,
                gate_stride[0] as i64,
                gate_stride[1] as i64,
                gate_stride[2] as i64,
                gate_stride[3] as i64,
                eps as f32,
                activation.sigmoid_flag(),
                dtype_code,
                stream,
            );
        }

        let output_storage = candle::RocmStorage::wrap_rocm_slice(output_buf, dev.clone());
        Ok(Tensor::from((
            candle::Storage::Rocm(output_storage),
            x.shape().clone(),
        )))
    }

    match x.dtype() {
        DType::F16 => rocm_fwd::<half::f16>(x, gate, weight, eps, activation, 0),
        DType::BF16 => rocm_fwd::<half::bf16>(x, gate, weight, eps, activation, 1),
        other => candle_core::bail!("rmsnorm_gated_rocm only supports f16/bf16, got {:?}", other),
    }
}

pub fn fused_gdn_gating_rocm(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use candle_core as candle;
    use core::ffi::c_void;

    fn rocm_fwd<T: candle::rocm_backend::RocmDType + candle::rocm_backend::hip::DeviceRepr>(
        b: &Tensor,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor)> {
        let total_elements = b.elem_count();
        let num_heads = a_log.elem_count();
        let shape = b.shape().clone();
        let dev = b.device().as_rocm_device()?;

        let (b_s, b_l) = b.storage_and_layout();
        let b_s = match &*b_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
            _ => candle::bail!("b must be a rocm tensor"),
        };
        let b_offset = b_l.start_offset();

        let (a_s, a_l) = a.storage_and_layout();
        let a_s = match &*a_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<T>()?,
            _ => candle::bail!("a must be a rocm tensor"),
        };
        let a_offset = a_l.start_offset();

        let (alog_s, alog_l) = a_log.storage_and_layout();
        let alog_s = match &*alog_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<f32>()?,
            _ => candle::bail!("a_log must be a rocm tensor"),
        };
        let alog_offset = alog_l.start_offset();

        let (dtb_s, dtb_l) = dt_bias.storage_and_layout();
        let dtb_s = match &*dtb_s {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<f32>()?,
            _ => candle::bail!("dt_bias must be a rocm tensor"),
        };
        let dtb_offset = dtb_l.start_offset();

        let beta_buf = unsafe { dev.alloc::<T>(total_elements) }?;
        let g_buf = unsafe { dev.alloc::<T>(total_elements) }?;

        let stream = dev.hip_stream().raw() as i64;

        unsafe {
            crate::rocm::ffi::fused_gdn_gating(
                b_s.slice(b_offset..).device_ptr() as *const c_void,
                a_s.slice(a_offset..).device_ptr() as *const c_void,
                alog_s.slice(alog_offset..).device_ptr() as *const f32,
                dtb_s.slice(dtb_offset..).device_ptr() as *const f32,
                beta_buf.device_ptr() as *mut c_void,
                g_buf.device_ptr() as *mut c_void,
                total_elements as i32,
                num_heads as i32,
                dtype_code,
                stream,
            );
        }

        let beta_storage = candle::RocmStorage::wrap_rocm_slice(beta_buf, dev.clone());
        let beta = Tensor::from((candle::Storage::Rocm(beta_storage), shape.clone()));

        let g_storage = candle::RocmStorage::wrap_rocm_slice(g_buf, dev.clone());
        let g = Tensor::from((candle::Storage::Rocm(g_storage), shape));

        Ok((beta, g))
    }

    match b.dtype() {
        DType::F16 => rocm_fwd::<half::f16>(b, a, a_log, dt_bias, 0),
        DType::BF16 => rocm_fwd::<half::bf16>(b, a, a_log, dt_bias, 1),
        other => candle_core::bail!(
            "fused_gdn_gating_rocm only supports f16/bf16, got {:?}",
            other
        ),
    }
}

/// Largest key head dimension the baseline decode kernel supports.
const GDN_DECODE_MAX_K: usize = 256;
/// `GdnDecodeKernel::Baseline` in gdn.cu: the key-major kernel without the
/// NVIDIA-specific cooperative and pipelined variants.
const GDN_DECODE_KERNEL_BASELINE: i32 = 0;

/// The decode recurrence kernel variant (gdn.cu's kernel kinds): the baseline unless
/// `MISTRALRS_GDN_DECODE_KERNEL` names `cooperative` or `pipelined`, which the ROCm build
/// also compiles.
fn decode_kernel_kind() -> i32 {
    static KIND: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *KIND.get_or_init(|| {
        match std::env::var("MISTRALRS_GDN_DECODE_KERNEL")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "cooperative" => 1,
            "pipelined" => 2,
            _ => GDN_DECODE_KERNEL_BASELINE,
        }
    })
}


/// Single-token GDN step from the convolved projections: gating, L2 norm and the
/// key-major recurrence in one kernel. Returns `[BH, 1, V]` in the input dtype.
/// Tokens per sequence the fused multi-token recurrence takes (speculative verify steps).
pub const GDN_FUSED_MULTI_MAX_TOKENS: usize = 8;

/// The fused decode recurrence over `seq_len` tokens per sequence, state in shared memory
/// between tokens. `mixed_qkv: [batch, seq_len, conv_dim]`, `b`/`a: [batch, seq_len, heads]`;
/// returns `[batch * heads, seq_len, head_v_dim]`.
pub fn fused_multi_recurrence_rocm(
    launch: FusedDecodeRecurrence<'_>,
    seq_len: usize,
) -> Result<Tensor> {
    use candle_core as candle;
    use core::ffi::c_void;

    let FusedDecodeRecurrence {
        mixed_qkv,
        b,
        a,
        a_log,
        dt_bias,
        state,
        batch_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tiled_v_heads,
        state_layout,
        slots,
    } = launch;
    if state_layout != RecurrentStateLayout::GdnKeyMajor || !matches!(head_k_dim, 64 | 128) {
        candle::bail!("ROCm fused multi-token GDN needs a key-major state with K of 64 or 128");
    }
    if b.rank() != 3 || a.rank() != 3 || mixed_qkv.dims() != [batch_size, seq_len, 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim] {
        candle::bail!("ROCm fused multi-token GDN: unexpected input shapes");
    }
    let dtype_code = match mixed_qkv.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        other => candle::bail!("fused_multi_recurrence_rocm only supports f16/bf16, got {other:?}"),
    };
    let dev = mixed_qkv.device().as_rocm_device()?;
    let half_ptr = |t: &Tensor, name: &str| -> Result<u64> {
        match t.dtype() {
            DType::F16 => rocm_ptr::<half::f16>(t, name),
            DType::BF16 => rocm_ptr::<half::bf16>(t, name),
            other => candle::bail!("{name} has unsupported dtype {other:?}"),
        }
    };
    let mixed_ptr = half_ptr(&mixed_qkv.contiguous()?, "mixed_qkv")?;
    let b_ptr = half_ptr(b, "b")?;
    let a_ptr = half_ptr(a, "a")?;
    let b_strides = b.stride().iter().map(|&s| s as i64).collect::<Vec<_>>();
    let a_strides = a.stride().iter().map(|&s| s as i64).collect::<Vec<_>>();
    let alog_ptr = rocm_ptr::<f32>(a_log, "a_log")?;
    let dtb_ptr = rocm_ptr::<f32>(dt_bias, "dt_bias")?;
    let (state_ptr, state_dtype) = rocm_recurrent_state_ptr(state, "state")?;
    let bh = batch_size * num_v_heads;
    let output = unsafe { Tensor::empty((bh, seq_len, head_v_dim), mixed_qkv.dtype(), mixed_qkv.device()) }?;
    let out_ptr = half_ptr(&output, "output")?;
    let stream = dev.hip_stream().raw() as i64;
    with_slot_indices(slots, |slot_ptr, _| {
        unsafe {
            crate::rocm::ffi::gdn_decode_recurrence_multi(
                mixed_ptr as *const c_void,
                b_ptr as *const c_void,
                a_ptr as *const c_void,
                alog_ptr as *const f32,
                dtb_ptr as *const f32,
                state_ptr,
                out_ptr as *mut c_void,
                batch_size as i32,
                seq_len as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                i32::from(tiled_v_heads),
                b_strides.as_ptr(),
                a_strides.as_ptr(),
                slot_ptr,
                dtype_code,
                state_dtype,
                stream,
            );
        }
        Ok(())
    })?;
    Ok(output)
}

pub fn fused_decode_recurrence_rocm(launch: FusedDecodeRecurrence<'_>) -> Result<Tensor> {
    use candle_core as candle;
    use core::ffi::c_void;

    let FusedDecodeRecurrence {
        mixed_qkv,
        b,
        a,
        a_log,
        dt_bias,
        state,
        batch_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tiled_v_heads,
        state_layout,
        slots,
    } = launch;
    if state_layout != RecurrentStateLayout::GdnKeyMajor || head_k_dim > GDN_DECODE_MAX_K {
        candle::bail!("ROCm GDN decode needs a key-major state with K <= {GDN_DECODE_MAX_K}");
    }
    let dtype_code = match mixed_qkv.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        other => candle::bail!("fused_decode_recurrence_rocm only supports f16/bf16, got {other:?}"),
    };
    let dev = mixed_qkv.device().as_rocm_device()?;
    let half_ptr = |t: &Tensor, name: &str| -> Result<u64> {
        match t.dtype() {
            DType::F16 => rocm_ptr::<half::f16>(t, name),
            DType::BF16 => rocm_ptr::<half::bf16>(t, name),
            other => candle::bail!("{name} has unsupported dtype {other:?}"),
        }
    };
    let mixed_ptr = half_ptr(mixed_qkv, "mixed_qkv")?;
    let b_ptr = half_ptr(b, "b")?;
    let a_ptr = half_ptr(a, "a")?;
    let b_stride = b.stride();
    let a_stride = a.stride();
    let alog_ptr = rocm_ptr::<f32>(a_log, "a_log")?;
    let dtb_ptr = rocm_ptr::<f32>(dt_bias, "dt_bias")?;
    let (state_ptr, state_dtype) = rocm_recurrent_state_ptr(state, "state")?;

    let bh = batch_size * num_v_heads;
    let output = unsafe { Tensor::empty((bh, 1, head_v_dim), mixed_qkv.dtype(), mixed_qkv.device()) }?;
    let out_ptr = half_ptr(&output, "output")?;
    let stream = dev.hip_stream().raw() as i64;
    with_slot_indices(slots, |slot_ptr, _| {
        unsafe {
            crate::rocm::ffi::gdn_decode_recurrence(
                mixed_ptr as *const c_void,
                b_ptr as *const c_void,
                a_ptr as *const c_void,
                alog_ptr as *const f32,
                dtb_ptr as *const f32,
                state_ptr,
                out_ptr as *mut c_void,
                batch_size as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                i32::from(tiled_v_heads),
                b_stride[0] as i64,
                b_stride[b_stride.len() - 1] as i64,
                a_stride[0] as i64,
                a_stride[a_stride.len() - 1] as i64,
                slot_ptr,
                decode_kernel_kind(),
                dtype_code,
                state_dtype,
                stream,
            );
        }
        Ok(())
    })?;
    Ok(output)
}

/// Splits the convolved projections into f32 `[BH, S, K]` q/k (L2 normalized),
/// `[BH, S, V]` v and `[BH, S]` gate/beta for the prefill recurrence.
#[allow(clippy::too_many_arguments)]
pub fn prepare_recurrence_inputs_rocm(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    batch_size: usize,
    seq_len: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    tiled_v_heads: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    use candle_core as candle;
    use core::ffi::c_void;

    let dtype_code = match mixed_qkv.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        other => candle::bail!("prepare_recurrence_inputs_rocm only supports f16/bf16, got {other:?}"),
    };
    let half_ptr = |t: &Tensor, name: &str| -> Result<u64> {
        match t.dtype() {
            DType::F16 => rocm_ptr::<half::f16>(t, name),
            DType::BF16 => rocm_ptr::<half::bf16>(t, name),
            other => candle::bail!("{name} has unsupported dtype {other:?}"),
        }
    };
    let dev = mixed_qkv.device();
    let bh = batch_size * num_v_heads;
    let empty = |shape: &[usize]| unsafe { Tensor::empty(shape, DType::F32, dev) };
    let q = empty(&[bh, seq_len, head_k_dim])?;
    let k = empty(&[bh, seq_len, head_k_dim])?;
    let v = empty(&[bh, seq_len, head_v_dim])?;
    let g = empty(&[bh, seq_len])?;
    let beta = empty(&[bh, seq_len])?;
    unsafe {
        crate::rocm::ffi::gdn_prepare_recurrence(
            half_ptr(mixed_qkv, "mixed_qkv")? as *const c_void,
            half_ptr(b, "b")? as *const c_void,
            half_ptr(a, "a")? as *const c_void,
            rocm_ptr::<f32>(a_log, "a_log")? as *const f32,
            rocm_ptr::<f32>(dt_bias, "dt_bias")? as *const f32,
            rocm_ptr::<f32>(&q, "q")? as *mut c_void,
            rocm_ptr::<f32>(&k, "k")? as *mut c_void,
            rocm_ptr::<f32>(&v, "v")? as *mut c_void,
            rocm_ptr::<f32>(&g, "g")? as *mut f32,
            rocm_ptr::<f32>(&beta, "beta")? as *mut f32,
            batch_size as i32,
            seq_len as i32,
            seq_len as i32,
            num_k_heads as i32,
            num_v_heads as i32,
            head_k_dim as i32,
            head_v_dim as i32,
            i32::from(tiled_v_heads),
            dtype_code,
            0,
            dev.as_rocm_device()?.hip_stream().raw() as i64,
        );
    }
    Ok((q, k, v, g, beta))
}
