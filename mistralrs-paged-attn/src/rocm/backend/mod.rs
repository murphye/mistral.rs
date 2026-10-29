mod cache;
mod gather_kv;
mod paged_attention;
mod scale_update;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::rocm_backend::hip::HipSlice;
use candle_core::{Layout, Result};
pub use gather_kv::gather_kv_cache;
pub use paged_attention::{paged_attention, paged_attention_with_window, reshape_and_cache};
pub use scale_update::kv_scale_update;

fn cache_input_layout(
    layout: &Layout,
    name: &str,
    op: &str,
) -> Result<(usize, usize, usize, usize)> {
    let (num_tokens, num_heads, head_size, row_stride) = match *layout.dims() {
        [num_tokens, num_heads, head_size] => {
            (num_tokens, num_heads, head_size, layout.stride()[0])
        }
        [batch, seq_len, num_heads, head_size] => {
            let num_tokens = batch
                .checked_mul(seq_len)
                .ok_or_else(|| candle_core::Error::msg("cache input token count overflow"))?;
            let row_stride = if seq_len == 1 {
                layout.stride()[0]
            } else {
                layout.stride()[1]
            };
            if batch > 1 && seq_len > 1 && layout.stride()[0] != seq_len.saturating_mul(row_stride)
            {
                candle_core::bail!("{op} cannot flatten {name} batch/sequence strides: {layout:?}");
            }
            (num_tokens, num_heads, head_size, row_stride)
        }
        _ => candle_core::bail!("{op} expects rank-3 or rank-4 {name} input, got {layout:?}"),
    };
    if layout.stride()[layout.stride().len() - 1] != 1
        || layout.stride()[layout.stride().len() - 2] != head_size
        || row_stride < num_heads.saturating_mul(head_size)
    {
        candle_core::bail!("{op} expects dense {name} heads, got {layout:?}");
    }
    Ok((num_tokens, num_heads, head_size, row_stride))
}

/// Device address of element `lo` of `v`. The unit second element mirrors the
/// `(ptr, guard)` pairs of the CUDA backend; HIP buffers need no event guard.
pub fn slice_ptr<T>(v: &HipSlice<T>, lo: usize) -> (u64, ()) {
    (v.device_ptr() + (lo * std::mem::size_of::<T>()) as u64, ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_input_layout_flattens_uniform_rank_four_rows() -> Result<()> {
        let prefill = Layout::new((2, 3, 2, 4).into(), vec![48, 16, 4, 1], 7);
        assert_eq!(cache_input_layout(&prefill, "key", "test")?, (6, 2, 4, 16));

        let decode = Layout::new((8, 1, 2, 4).into(), vec![24, 8, 4, 1], 5);
        assert_eq!(cache_input_layout(&decode, "value", "test")?, (8, 2, 4, 24));
        Ok(())
    }

    /// Tokens written into the paged cache come back unchanged from the gather that
    /// continued prefills read their cached prefix with.
    #[test]
    fn gathered_cache_rows_match_written_rows() -> Result<()> {
        use candle_core::{DType, Device, Tensor};
        let Ok(device) = Device::new_rocm(0) else {
            return Ok(());
        };
        let (block_size, head_size, n) = (32, 256, 51);
        for kv_heads in [1, 2] {
            let x = 8; // bf16 elements per 16 bytes
            let num_blocks = 8;
            let key_cache = Tensor::zeros(
                (num_blocks, kv_heads, head_size / x, block_size, x),
                DType::BF16,
                &device,
            )?;
            let value_cache =
                Tensor::zeros((num_blocks, kv_heads, head_size, block_size), DType::BF16, &device)?;
            let make = |salt: f32| -> Result<Tensor> {
                Tensor::from_vec(
                    (0..n * kv_heads * head_size)
                        .map(|i| ((i as f32 * 0.013 + salt).sin()))
                        .collect::<Vec<_>>(),
                    (n, kv_heads, head_size),
                    &Device::Cpu,
                )?
                .to_dtype(DType::BF16)?
                .to_device(&device)
            };
            let (key, value) = (make(0.3)?, make(1.7)?);
            // The sequence occupies blocks 5 then 2.
            let blocks = [5i64, 2];
            let slots = (0..n as i64)
                .map(|t| blocks[(t / block_size as i64) as usize] * block_size as i64 + t % block_size as i64)
                .collect::<Vec<_>>();
            let slot_mapping = Tensor::from_vec(slots, n, &device)?;
            reshape_and_cache(&key, &value, None, None, &key_cache, &value_cache, &slot_mapping)?;
            let block_table = Tensor::from_vec(vec![5u32, 2], (1, 2), &device)?;
            let cu = Tensor::from_vec(vec![0u32, n as u32], 2, &device)?;
            let (k, v) = gather_kv_cache(
                &key_cache, &value_cache, None, None, &block_table, &cu, n, DType::BF16,
            )?;
            for (name, got, want) in [("key", &k, &key), ("value", &v, &value)] {
                let err = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert_eq!(err, 0.0, "{name} rows differ after gather with {kv_heads} kv heads");
            }
        }
        Ok(())
    }
}
