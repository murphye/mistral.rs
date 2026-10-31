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
    /// 64-token chunks, scalar math.
    Chunked,
    /// 64-token chunks on the matrix cores (`gdn_chunked_wmma.cu`); 128/128 head dims only.
    ChunkedWmma,
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
    if matches!(kernel, RecurrenceKernel::ChunkedWmma) {
        if k_dim != 128 || v_dim != 128 {
            candle_core::bail!(
                "the chunked WMMA GDN recurrence needs 128/128 head dims, got {k_dim}/{v_dim}"
            );
        }
        let (bh_i, seq_i) = (bh as i32, seq_len as i32);
        let tile_bytes =
            unsafe { crate::rocm::ffi::chunked_wmma_gdn_tile_bytes(bh_i, seq_i) } as usize;
        let snap_bytes =
            unsafe { crate::rocm::ffi::chunked_wmma_gdn_snapshot_bytes(bh_i, seq_i) } as usize;
        let chunks = unsafe { crate::rocm::ffi::chunked_wmma_gdn_num_chunks(seq_i) } as usize;
        // Pass-1 products (W, U, K'), the deltas, the per-chunk state snapshots and decays; all
        // transient, released with the stream's block cache after the launch.
        let w = unsafe { dev.alloc::<u8>(tile_bytes) }?;
        let u = unsafe { dev.alloc::<u8>(tile_bytes) }?;
        let kp = unsafe { dev.alloc::<u8>(tile_bytes) }?;
        let dlt = unsafe { dev.alloc::<u8>(tile_bytes) }?;
        let snap = unsafe { dev.alloc::<u8>(snap_bytes) }?;
        let gtot = unsafe { dev.alloc::<f32>(bh * chunks) }?;
        with_slot_indices(slots, |slot_ptr, batch| {
            let num_heads = bh.checked_div(batch).unwrap_or(1);
            let launched = unsafe {
                crate::rocm::ffi::chunked_wmma_gated_delta_rule_recurrence(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    g_ptr,
                    beta_ptr,
                    state_ptr,
                    output_buf.device_ptr() as *mut f32,
                    bh_i,
                    seq_i,
                    k_dim as i32,
                    v_dim as i32,
                    slot_ptr,
                    num_heads as i32,
                    state_dtype,
                    w.device_ptr() as *mut core::ffi::c_void,
                    u.device_ptr() as *mut core::ffi::c_void,
                    kp.device_ptr() as *mut core::ffi::c_void,
                    dlt.device_ptr() as *mut core::ffi::c_void,
                    snap.device_ptr() as *mut core::ffi::c_void,
                    gtot.device_ptr() as *mut f32,
                    stream,
                )
            };
            if launched == 0 {
                candle_core::bail!("the chunked WMMA GDN recurrence rejected the launch shape");
            }
            Ok(())
        })?;
        let output_storage = candle::RocmStorage::wrap_rocm_slice(output_buf, dev.clone());
        return Ok(Tensor::from((
            candle::Storage::Rocm(output_storage),
            (bh, seq_len, v_dim),
        )));
    }
    let launcher = match kernel {
        RecurrenceKernel::Scalar => crate::rocm::ffi::gated_delta_rule_recurrence,
        RecurrenceKernel::Warp => crate::rocm::ffi::warp_gated_delta_rule_recurrence,
        RecurrenceKernel::Chunked => crate::rocm::ffi::chunked_gated_delta_rule_recurrence,
        RecurrenceKernel::ChunkedWmma => unreachable!("handled above"),
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

/// Speculative GDN commit on ROCm: for every sequence with a nonzero keep count, the conv and
/// recurrent states after its first `keep_rows` verified rows, rebuilt from the pre-step
/// snapshot and the stashed step inputs and written to its pool slot (gdn.cu's
/// `gdn_speculative_state_commit`, one launch per layer for the whole batch).
pub fn speculative_state_commit_rocm(
    commit: crate::cuda::gdn::GdnSpeculativeStateCommit<'_>,
) -> Result<()> {
    use candle_core as candle;
    use core::ffi::c_void;

    let crate::cuda::gdn::GdnSpeculativeStateCommit {
        mixed_qkv,
        convolved_qkv,
        b,
        a,
        initial_conv_state,
        initial_recurrent_state,
        a_log,
        dt_bias,
        conv_state_pool,
        recurrent_state_pool,
        keep_rows,
        slot_indices,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tiled_v_heads,
        state_layout,
    } = commit;
    let (batch_size, seq_len, _) = mixed_qkv.dims3()?;
    let kernel_size = initial_conv_state.dim(candle_core::D::Minus1)?;
    let value_major = match state_layout {
        RecurrentStateLayout::GdnKeyMajor => false,
        RecurrentStateLayout::GdnValueMajor => true,
        RecurrentStateLayout::Opaque => {
            candle::bail!("GDN speculative commit does not support opaque state")
        }
    };
    if keep_rows.dims1()? != batch_size || slot_indices.dims1()? != batch_size {
        candle::bail!("GDN speculative commit index tensors must match batch size");
    }
    if !conv_state_pool.is_contiguous() || !recurrent_state_pool.is_contiguous() {
        candle::bail!("GDN speculative commit state pools must be contiguous");
    }
    let dtype_code = match mixed_qkv.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        other => {
            candle::bail!("GDN speculative state commit only supports f16/bf16, got {other:?}")
        }
    };
    let half_ptr = |t: &Tensor, name: &str| -> Result<*mut c_void> {
        Ok(match t.dtype() {
            DType::F16 => rocm_ptr::<half::f16>(t, name)?,
            DType::BF16 => rocm_ptr::<half::bf16>(t, name)?,
            other => candle::bail!("{name} has unsupported dtype {other:?}"),
        } as *mut c_void)
    };
    let mixed_qkv = mixed_qkv.contiguous()?;
    let convolved_qkv = convolved_qkv.contiguous()?;
    let b = b.contiguous()?;
    let a = a.contiguous()?;
    let initial_conv_state = initial_conv_state.contiguous()?;
    let initial_recurrent_state = initial_recurrent_state.contiguous()?;
    let a_log = a_log.to_dtype(DType::F32)?.contiguous()?;
    let dt_bias = dt_bias.to_dtype(DType::F32)?.contiguous()?;
    let (initial_recurrent_ptr, state_dtype) =
        rocm_recurrent_state_ptr(&initial_recurrent_state, "initial_recurrent_state")?;
    let (recurrent_pool_ptr, pool_state_dtype) =
        rocm_recurrent_state_ptr(recurrent_state_pool, "recurrent_state_pool")?;
    if state_dtype != pool_state_dtype {
        candle::bail!("GDN speculative commit recurrent state dtype mismatch");
    }
    let stream = mixed_qkv.device().as_rocm_device()?.hip_stream().raw() as i64;
    unsafe {
        crate::rocm::ffi::gdn_speculative_state_commit(
            half_ptr(&mixed_qkv, "mixed_qkv")?,
            half_ptr(&convolved_qkv, "convolved_qkv")?,
            half_ptr(&b, "b")?,
            half_ptr(&a, "a")?,
            half_ptr(&initial_conv_state, "initial_conv_state")?,
            initial_recurrent_ptr,
            rocm_ptr::<f32>(&a_log, "a_log")? as *const f32,
            rocm_ptr::<f32>(&dt_bias, "dt_bias")? as *const f32,
            half_ptr(conv_state_pool, "conv_state_pool")?,
            recurrent_pool_ptr,
            rocm_ptr::<u32>(keep_rows, "keep_rows")? as *const u32,
            rocm_ptr::<u32>(slot_indices, "slot_indices")? as *const u32,
            batch_size as i32,
            seq_len as i32,
            num_k_heads as i32,
            num_v_heads as i32,
            head_k_dim as i32,
            head_v_dim as i32,
            kernel_size as i32,
            i32::from(tiled_v_heads),
            i32::from(value_major),
            dtype_code,
            state_dtype,
            stream,
        );
    }
    Ok(())
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
    seed_k: Option<usize>,
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
        seed_k: Option<usize>,
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
        // With `seed_k` (the consuming projection's input width) the kernel also writes the
        // output's q8_1 form, recorded as the stream's shared activation quantization.
        let seed_tokens = seed_k
            .filter(|k| {
                dtype_code == 1
                    && hidden_dim == 128
                    && *k % 512 == 0
                    && (rows * hidden_dim) % *k == 0
            })
            .map(|k| (k, rows * hidden_dim / k));
        let q8 = seed_tokens
            .map(|_| unsafe { dev.alloc::<u8>(rows * (hidden_dim / 32) * 36) })
            .transpose()?;
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
                q8.as_ref().map_or(std::ptr::null_mut(), |q8| q8.device_ptr() as *mut c_void),
                dtype_code,
                stream,
            );
        }
        if let (Some(q8), Some((k, tokens))) = (q8, seed_tokens) {
            // The key candle's SharedQ8Activation computes for bf16 rows of this shape.
            dev.hip_stream()
                .set_derived((output_buf.device_ptr(), k, tokens * 4), q8);
        }

        let output_storage = candle::RocmStorage::wrap_rocm_slice(output_buf, dev.clone());
        Ok(Tensor::from((
            candle::Storage::Rocm(output_storage),
            x.shape().clone(),
        )))
    }

    match x.dtype() {
        DType::F16 => rocm_fwd::<half::f16>(x, gate, weight, eps, activation, 0, seed_k),
        DType::BF16 => rocm_fwd::<half::bf16>(x, gate, weight, eps, activation, 1, seed_k),
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

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::*;

    fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as u16 as f32;
                ((x / 128.0) - 1.0) * scale + offset
            })
            .collect()
    }

    fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
        tensor
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, |m, d| if d.is_nan() { f32::NAN } else { m.max(d) })
    }

    /// The fused speculative commit against a replay of each sequence's kept rows through the
    /// separate conv and verify-recurrence kernels and against the recurrence on the host:
    /// conv states exactly, recurrent states closely, and slots of sequences that keep nothing
    /// untouched.
    #[test]
    fn speculative_state_commit_matches_prefix_replay_rocm() -> Result<()> {
        let Ok(dev) = Device::new_rocm(0) else {
            return Ok(());
        };
        let (batch, seq_len, kernel_size, capacity) = (3usize, 4usize, 4usize, 5usize);
        let (num_k_heads, num_v_heads, head_k_dim, head_v_dim) =
            (1usize, 2usize, 128usize, 128usize);
        let conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
        let row_state = num_v_heads * head_k_dim * head_v_dim;
        let row_conv = conv_dim * kernel_size;
        let keep_rows = vec![1u32, 3, 0];
        let slots = vec![4u32, 1, 3];
        let bf16 = |data: Vec<f32>, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_vec(data, shape, &dev)?.to_dtype(DType::BF16)
        };
        let mixed_qkv = bf16(
            patterned(batch * seq_len * conv_dim, 110, 0.08, 0.01),
            &[batch, seq_len, conv_dim],
        )?;
        let conv_weight = bf16(
            patterned(conv_dim * kernel_size, 111, 0.05, -0.01),
            &[conv_dim, kernel_size],
        )?;
        let initial_conv = bf16(
            patterned(batch * row_conv, 112, 0.03, 0.0),
            &[batch, conv_dim, kernel_size],
        )?;
        let (convolved, _) = causal_conv1d_rocm(
            &mixed_qkv,
            &conv_weight,
            &initial_conv,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        let b = bf16(
            patterned(batch * seq_len * num_v_heads, 113, 0.2, 0.1),
            &[batch, seq_len, num_v_heads],
        )?;
        let a = bf16(
            patterned(batch * seq_len * num_v_heads, 114, 0.18, -0.04),
            &[batch, seq_len, num_v_heads],
        )?;
        let a_log = Tensor::from_vec(patterned(num_v_heads, 115, 0.05, -0.2), num_v_heads, &dev)?;
        let dt_bias = Tensor::from_vec(patterned(num_v_heads, 116, 0.1, 0.3), num_v_heads, &dev)?;
        for state_dtype in [DType::F32, DType::BF16] {
            let initial_state = Tensor::from_vec(
                patterned(batch * row_state, 117, 0.02, 0.0),
                (batch, num_v_heads, head_k_dim, head_v_dim),
                &dev,
            )?
            .to_dtype(state_dtype)?;
            let conv_pool = bf16(
                patterned(capacity * row_conv, 118, 0.04, 0.0),
                &[capacity, conv_dim, kernel_size],
            )?;
            let state_pool = Tensor::from_vec(
                patterned(capacity * row_state, 119, 0.02, 0.0),
                (capacity, num_v_heads, head_k_dim, head_v_dim),
                &dev,
            )?
            .to_dtype(state_dtype)?;
            let mut want_conv = flat(&conv_pool)?;
            let mut want_state = flat(&state_pool)?;
            for (i, &rows) in keep_rows.iter().enumerate() {
                let rows = rows as usize;
                if rows == 0 {
                    continue;
                }
                let (_, conv_state) = causal_conv1d_rocm(
                    &mixed_qkv
                        .narrow(0, i, 1)?
                        .narrow(1, 0, rows)?
                        .contiguous()?,
                    &conv_weight,
                    &initial_conv.narrow(0, i, 1)?.contiguous()?,
                    kernel_size,
                    false,
                    GdnStateSlots::Gathered,
                )?;
                let dst = slots[i] as usize * row_conv;
                want_conv[dst..dst + row_conv].copy_from_slice(&flat(&conv_state)?);
                // The replay this commit replaces: the verify step's own recurrence kernels
                // over the kept rows, from the pre-step state.
                // A copy: the kernels update the state in place, and an f32 narrow would alias
                // the initial state the commit reads.
                let mut state = initial_state
                    .narrow(0, i, 1)?
                    .to_dtype(DType::F32)?
                    .reshape((num_v_heads, head_k_dim, head_v_dim))?
                    .copy()?;
                let (convolved_row, b_row, a_row) = (
                    convolved
                        .narrow(0, i, 1)?
                        .narrow(1, 0, rows)?
                        .contiguous()?,
                    b.narrow(0, i, 1)?.narrow(1, 0, rows)?.contiguous()?,
                    a.narrow(0, i, 1)?.narrow(1, 0, rows)?.contiguous()?,
                );
                let launch = FusedDecodeRecurrence {
                    mixed_qkv: &convolved_row,
                    b: &b_row,
                    a: &a_row,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    state: &mut state,
                    batch_size: 1,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    tiled_v_heads: false,
                    state_layout: RecurrentStateLayout::GdnKeyMajor,
                    slots: GdnStateSlots::Gathered,
                };
                if rows == 1 {
                    fused_decode_recurrence_rocm(launch)?;
                } else {
                    fused_multi_recurrence_rocm(launch, rows)?;
                }
                let dst = slots[i] as usize * row_state;
                want_state[dst..dst + row_state].copy_from_slice(&flat(&state)?);
            }
            speculative_state_commit_rocm(crate::cuda::gdn::GdnSpeculativeStateCommit {
                mixed_qkv: &mixed_qkv,
                convolved_qkv: &convolved,
                b: &b,
                a: &a,
                initial_conv_state: &initial_conv,
                initial_recurrent_state: &initial_state,
                a_log: &a_log,
                dt_bias: &dt_bias,
                conv_state_pool: &conv_pool,
                recurrent_state_pool: &state_pool,
                keep_rows: &Tensor::from_vec(keep_rows.clone(), batch, &dev)?,
                slot_indices: &Tensor::from_vec(slots.clone(), batch, &dev)?,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                tiled_v_heads: false,
                state_layout: RecurrentStateLayout::GdnKeyMajor,
            })?;
            // The gated delta rule on the host from the same convolved inputs.
            let conv_h = flat(&convolved)?;
            let (b_h, a_h) = (flat(&b)?, flat(&a)?);
            let (alog_h, dtb_h) = (flat(&a_log)?, flat(&dt_bias)?);
            let init_h = flat(&initial_state)?;
            let mut cpu_state = flat(&state_pool)?;
            let key_dim = num_k_heads * head_k_dim;
            for (i, &rows) in keep_rows.iter().enumerate() {
                for vh in 0..num_v_heads {
                    let kh = vh / (num_v_heads / num_k_heads);
                    let base = (i * num_v_heads + vh) * head_k_dim * head_v_dim;
                    let mut st = init_h[base..base + head_k_dim * head_v_dim].to_vec();
                    for p in 0..rows as usize {
                        let row =
                            &conv_h[(i * seq_len + p) * conv_dim..(i * seq_len + p + 1) * conv_dim];
                        let key = &row[key_dim + kh * head_k_dim..key_dim + (kh + 1) * head_k_dim];
                        let inv = 1.0 / (key.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                        let gi = (i * seq_len + p) * num_v_heads + vh;
                        let beta = 1.0 / (1.0 + (-b_h[gi]).exp());
                        let av = a_h[gi] + dtb_h[vh];
                        let sp = if av > 20.0 { av } else { av.exp().ln_1p() };
                        let decay = (-alog_h[vh].exp() * sp).exp();
                        for vi in 0..head_v_dim {
                            let v = row[2 * key_dim + vh * head_v_dim + vi];
                            let dot: f32 = (0..head_k_dim)
                                .map(|k| st[k * head_v_dim + vi] * key[k] * inv)
                                .sum();
                            let delta = (v - decay * dot) * beta;
                            for k in 0..head_k_dim {
                                st[k * head_v_dim + vi] =
                                    key[k] * inv * delta + decay * st[k * head_v_dim + vi];
                            }
                        }
                    }
                    if rows > 0 {
                        let dst = (slots[i] as usize * num_v_heads + vh) * head_k_dim * head_v_dim;
                        cpu_state[dst..dst + st.len()].copy_from_slice(&st);
                    }
                }
            }
            if state_dtype == DType::F32 {
                let cpu_diff = max_abs_diff(&flat(&state_pool)?, &cpu_state);
                assert!(
                    cpu_diff <= 1e-6,
                    "recurrent state vs the CPU recurrence: max diff {cpu_diff}"
                );
            }
            let conv_diff = max_abs_diff(&flat(&conv_pool)?, &want_conv);
            assert!(
                conv_diff == 0.0,
                "{state_dtype:?} conv state: max diff {conv_diff}"
            );
            // The replay keeps its state in f32; a bf16 pool rounds the committed state once.
            let tol = if state_dtype == DType::F32 {
                2e-4
            } else {
                4e-3
            };
            let state_diff = max_abs_diff(&flat(&state_pool)?, &want_state);
            assert!(
                state_diff <= tol,
                "{state_dtype:?} recurrent state: max diff {state_diff}"
            );
        }
        Ok(())
    }
}
