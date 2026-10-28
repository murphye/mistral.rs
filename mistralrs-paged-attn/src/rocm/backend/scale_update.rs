use crate::rocm::backend::slice_ptr;
use crate::rocm::ffi;
use candle_core as candle;
use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
struct KvScaleUpdate {
    k_scales: Tensor,
    v_scales: Tensor,
}

impl candle::InplaceOp2 for KvScaleUpdate {
    fn name(&self) -> &'static str {
        "kvscale-update"
    }

    fn cpu_fwd(
        &self,
        _: &mut candle::CpuStorage,
        _: &candle::Layout,
        _: &candle::CpuStorage,
        _: &candle::Layout,
    ) -> Result<()> {
        panic!("kvscale-update is not implemented on CPU!")
    }

    fn rocm_fwd(
        &self,
        k: &mut candle::RocmStorage,
        k_layout: &candle::Layout,
        v: &candle::RocmStorage,
        _: &candle::Layout,
    ) -> Result<()> {
        use candle::backend::BackendStorage;
        use candle::rocm_backend::RocmStorageSlice;
        let dev = k.device();
        let elem_count = k_layout.shape().elem_count();

        use std::ffi::c_void;

        let (src_ptr, dst_ptr) = match (&k.slice, &v.slice) {
            (RocmStorageSlice::BF16(inp_k), RocmStorageSlice::BF16(inp_v)) => (
                inp_k.device_ptr() as *const c_void,
                inp_v.device_ptr() as *const c_void,
            ),
            (RocmStorageSlice::F16(inp_k), RocmStorageSlice::F16(inp_v)) => (
                inp_k.device_ptr() as *const c_void,
                inp_v.device_ptr() as *const c_void,
            ),
            (RocmStorageSlice::F32(inp_k), RocmStorageSlice::F32(inp_v)) => (
                inp_k.device_ptr() as *const c_void,
                inp_v.device_ptr() as *const c_void,
            ),
            _ => {
                panic!("Invalid dtype for kv scale update!")
            }
        };
        let stream = dev.hip_stream().raw() as i64;

        let (k_scales, k_scales_layout) = self.k_scales.storage_and_layout();
        let k_scales = match &*k_scales {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<f32>()?,
            _ => candle::bail!("k_scales must be a rocm tensor"),
        };
        let (k_scales, _) = slice_ptr(k_scales, k_scales_layout.start_offset());

        let (v_scales, v_scales_layout) = self.v_scales.storage_and_layout();
        let v_scales = match &*v_scales {
            candle::Storage::Rocm(c) => c.as_rocm_slice::<f32>()?,
            _ => candle::bail!("v_scales must be a rocm tensor"),
        };
        let (v_scales, _) = slice_ptr(v_scales, v_scales_layout.start_offset());

        let k_scales_ptr = k_scales as *mut f32;
        let v_scales_ptr = v_scales as *mut f32;
        unsafe {
            match k.dtype() {
                DType::F32 => ffi::update_kv_scales_f32(
                    src_ptr,
                    dst_ptr,
                    elem_count as i64,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                DType::F16 => ffi::update_kv_scales_f16(
                    src_ptr,
                    dst_ptr,
                    elem_count as i64,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                DType::BF16 => ffi::update_kv_scales_bf16(
                    src_ptr,
                    dst_ptr,
                    elem_count as i64,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                _ => {
                    panic!("Invalid dtype for kv scale update!")
                }
            }
        }
        Ok(())
    }
}

pub fn kv_scale_update(
    key: &Tensor,
    value: &Tensor,
    k_scales: &Tensor,
    v_scales: &Tensor,
) -> Result<()> {
    let op = KvScaleUpdate {
        k_scales: k_scales.to_owned(),
        v_scales: v_scales.to_owned(),
    };
    key.inplace_op2(value, &op)
}
