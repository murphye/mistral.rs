//! Large-vocabulary top-k for token sampling on the device.

use std::ffi::c_void;

use candle_core::{DType, Result, Tensor};

use crate::rocm::rocm_ptr;

const CHUNK_SIZE: usize = 2048;
pub(crate) const MAX_K: usize = 128;

/// Top-`k` logits of a single row, packed as `[values(k), indices(k), softmax_denominator,
/// global_max]` where the softmax terms are taken over all of `logits / temperature`.
pub fn topk_logits_packed(logits: &Tensor, k: usize, temperature: f64) -> Result<Vec<f32>> {
    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("topk_logits_packed requires a positive finite temperature");
    }
    let logits = logits.flatten_all()?.contiguous()?;
    let ncols = logits.elem_count();
    let k = k.min(ncols);
    if k == 0 || k > MAX_K {
        candle_core::bail!("topk_logits_packed: k={k} must be in [1, {MAX_K}]");
    }
    let nblocks = ncols.div_ceil(CHUNK_SIZE);
    let device = logits.device();
    let block_values = unsafe { Tensor::empty(nblocks * k, DType::F32, device) }?;
    let block_indices = unsafe { Tensor::empty(nblocks * k, DType::U32, device) }?;
    let block_maxes = unsafe { Tensor::empty(nblocks, DType::F32, device) }?;
    let block_sums = unsafe { Tensor::empty(nblocks, DType::F32, device) }?;
    let packed = unsafe { Tensor::empty(2 * k + 2, DType::F32, device) }?;
    let (input, launch) = match logits.dtype() {
        DType::F32 => (
            rocm_ptr::<f32>(&logits, "logits")?,
            crate::rocm::ffi::topk_large_packed_f32 as unsafe extern "C" fn(_, _, _, _, _, _, _, _, _, _, _, _),
        ),
        DType::F16 => (
            rocm_ptr::<half::f16>(&logits, "logits")?,
            crate::rocm::ffi::topk_large_packed_f16 as _,
        ),
        DType::BF16 => (
            rocm_ptr::<half::bf16>(&logits, "logits")?,
            crate::rocm::ffi::topk_large_packed_bf16 as _,
        ),
        dt => candle_core::bail!("topk_logits_packed: unsupported dtype {dt:?}"),
    };
    unsafe {
        launch(
            input as *const c_void,
            rocm_ptr::<f32>(&block_values, "block_values")? as *mut f32,
            rocm_ptr::<u32>(&block_indices, "block_indices")? as *mut u32,
            rocm_ptr::<f32>(&block_maxes, "block_maxes")? as *mut f32,
            rocm_ptr::<f32>(&block_sums, "block_sums")? as *mut f32,
            rocm_ptr::<f32>(&packed, "packed")? as *mut f32,
            ncols as i32,
            k as i32,
            CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            device.as_rocm_device()?.hip_stream().raw() as i64,
        );
    }
    packed.to_vec1::<f32>()
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    #[test]
    fn rocm_topk_logits_packed_matches_cpu() -> candle_core::Result<()> {
        let dev = Device::new_rocm(0)?;
        let (ncols, k, temperature) = (151_936usize, 20usize, 0.7f64);
        let logits = Tensor::randn(0f32, 3.0, ncols, &Device::Cpu)?;
        let host = logits.to_vec1::<f32>()?;
        let packed = super::topk_logits_packed(&logits.to_device(&dev)?, k, temperature)?;

        let mut order: Vec<usize> = (0..ncols).collect();
        order.sort_by(|a, b| host[*b].total_cmp(&host[*a]));
        for (i, &idx) in order.iter().take(k).enumerate() {
            assert_eq!(packed[k + i] as usize, idx);
            assert_eq!(packed[i], host[idx]);
        }
        let inv_t = (1.0 / temperature) as f32;
        let max = host[order[0]] * inv_t;
        let denom: f64 = host.iter().map(|v| ((v * inv_t - max) as f64).exp()).sum();
        assert_eq!(packed[2 * k + 1], max);
        assert!((packed[2 * k] as f64 - denom).abs() / denom < 1e-4);
        Ok(())
    }
}
