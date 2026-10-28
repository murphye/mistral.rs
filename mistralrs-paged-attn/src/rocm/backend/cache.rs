use std::{collections::HashMap, iter::zip, mem::size_of};

use crate::rocm::backend::slice_ptr;
use crate::rocm::ffi::{copy_blocks_bf16, copy_blocks_f16, copy_blocks_f32, copy_blocks_u8};
use candle_core::backend::BackendDevice;
use candle_core::rocm_backend::hip::{self, sys};
use candle_core::rocm_backend::RocmStorageSlice;
use candle_core::Result;
use candle_core::{Device, Storage, Tensor};

fn storage_ptr(slice: &RocmStorageSlice, offset: usize) -> Result<u64> {
    Ok(match slice {
        RocmStorageSlice::BF16(s) => slice_ptr(s, offset).0,
        RocmStorageSlice::F16(s) => slice_ptr(s, offset).0,
        RocmStorageSlice::F32(s) => slice_ptr(s, offset).0,
        RocmStorageSlice::F8E4M3(s) => slice_ptr(s, offset).0,
        _ => candle_core::bail!("only f32, f16, bf16 and f8e4m3 input data types are supported"),
    })
}

/// Copies whole cache blocks within each layer: `block_mapping` maps a source
/// block to the blocks it is copied to.
pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    if key_caches.is_empty() {
        if value_caches.is_empty() {
            return Ok(());
        }
        candle_core::bail!("copy_blocks requires the same number of key and value caches");
    }
    if key_caches.len() != value_caches.len() {
        candle_core::bail!("copy_blocks requires the same number of key and value caches");
    }
    if block_mapping.values().all(Vec::is_empty) {
        return Ok(());
    }

    let cache_dev = key_caches[0].device();
    let Device::Rocm(dev) = cache_dev else {
        candle_core::bail!("copy_blocks requires ROCm caches")
    };
    let dtype = key_caches[0].dtype();
    if key_caches[0].rank() == 0 || value_caches[0].rank() == 0 {
        candle_core::bail!("copy_blocks caches must have a block dimension");
    }
    let numel_per_block_key = key_caches[0].dims()[1..].iter().product::<usize>();
    let numel_per_block_value = value_caches[0].dims()[1..].iter().product::<usize>();
    let num_layers = i32::try_from(key_caches.len())?;

    let mut cache_ptrs = Vec::with_capacity(key_caches.len() * 2);
    let mut value_cache_ptrs = Vec::with_capacity(key_caches.len());
    for (layer, (key_cache, value_cache)) in zip(&key_caches, &value_caches).enumerate() {
        if !cache_dev.same_device(key_cache.device())
            || !cache_dev.same_device(value_cache.device())
        {
            candle_core::bail!("copy_blocks cache layer {layer} is on a different ROCm device");
        }
        if key_cache.dtype() != dtype || value_cache.dtype() != dtype {
            candle_core::bail!("copy_blocks cache layer {layer} has a mismatched dtype");
        }
        if !key_cache.is_contiguous() || !value_cache.is_contiguous() {
            candle_core::bail!("copy_blocks cache layer {layer} must be contiguous");
        }
        if key_cache.dims()[1..].iter().product::<usize>() != numel_per_block_key
            || value_cache.dims()[1..].iter().product::<usize>() != numel_per_block_value
        {
            candle_core::bail!("copy_blocks cache layer {layer} has a mismatched block shape");
        }
        for (src_block, dst_blocks) in block_mapping {
            let num_blocks = key_cache.dims()[0].min(value_cache.dims()[0]);
            if *src_block >= num_blocks || dst_blocks.iter().any(|dst| *dst >= num_blocks) {
                candle_core::bail!("copy_blocks block index out of bounds for cache layer {layer}");
            }
        }
        let (key_storage, key_layout) = key_cache.storage_and_layout();
        let (value_storage, value_layout) = value_cache.storage_and_layout();
        let (Storage::Rocm(key_storage), Storage::Rocm(value_storage)) =
            (&*key_storage, &*value_storage)
        else {
            candle_core::bail!("copy_blocks cache layer {layer} is not on ROCm");
        };
        cache_ptrs.push(storage_ptr(&key_storage.slice, key_layout.start_offset())? as i64);
        value_cache_ptrs
            .push(storage_ptr(&value_storage.slice, value_layout.start_offset())? as i64);
    }

    let mut block_mapping_vec: Vec<i64> = Vec::new();
    for (src_block_number, dst_blocks) in block_mapping {
        for dst_block_number in dst_blocks {
            block_mapping_vec.push(i64::try_from(*src_block_number)?);
            block_mapping_vec.push(i64::try_from(*dst_block_number)?);
        }
    }
    let num_pairs = i32::try_from(block_mapping_vec.len() / 2)?;

    // Key pointers, value pointers and the block mapping share one device buffer.
    // It is freed on the stream after the kernel that reads it.
    let value_ptr_offset = cache_ptrs.len();
    let mapping_offset = value_ptr_offset + value_cache_ptrs.len();
    cache_ptrs.extend(value_cache_ptrs);
    cache_ptrs.extend(block_mapping_vec);
    let metadata = dev.clone_htod(&cache_ptrs)?;
    let metadata_ptr = metadata.device_ptr();
    let key_cache_ptr = metadata_ptr as *mut core::ffi::c_void;
    let value_cache_ptr =
        (metadata_ptr + (value_ptr_offset * size_of::<i64>()) as u64) as *mut core::ffi::c_void;
    let block_mapping_ptr =
        (metadata_ptr + (mapping_offset * size_of::<i64>()) as u64) as *const core::ffi::c_void;
    let numel_per_block_key = i32::try_from(numel_per_block_key)?;
    let numel_per_block_value = i32::try_from(numel_per_block_value)?;
    let stream = dev.hip_stream();
    stream.bind().map_err(candle_core::Error::wrap)?;
    let launch = match dtype {
        candle_core::DType::BF16 => copy_blocks_bf16,
        candle_core::DType::F16 => copy_blocks_f16,
        candle_core::DType::F32 => copy_blocks_f32,
        candle_core::DType::F8E4M3 => copy_blocks_u8,
        _ => candle_core::bail!("only f32, f16, bf16 and f8e4m3 input data types are supported"),
    };
    unsafe {
        launch(
            key_cache_ptr,
            value_cache_ptr,
            block_mapping_ptr,
            num_layers,
            num_pairs,
            numel_per_block_key,
            numel_per_block_value,
            stream.raw() as i64,
        );
    }
    drop(metadata);
    Ok(())
}

/// Copies blocks between a CPU and a ROCm cache, or within one ROCm device.
///
/// # Safety
/// `dst` is the only shared reference and upholds the `&mut` aliasing guarantee.
pub unsafe fn swap_blocks(
    src: Tensor,
    dst: &Tensor,
    block_mapping: HashMap<usize, usize>,
) -> Result<()> {
    let block_size_in_bytes = src.dtype().size_in_bytes() * src.dims()[0];
    let (src_storage, src_layout) = src.storage_and_layout();
    let (dst_storage, dst_layout) = dst.storage_and_layout();
    let (dst_ptr, dst_dev) = match &*dst_storage {
        Storage::Rocm(s) => (
            storage_ptr(&s.slice, dst_layout.start_offset())?,
            s.device.clone(),
        ),
        _ => candle_core::bail!("swap_blocks destination must be a ROCm tensor"),
    };
    let stream = dst_dev.hip_stream();
    stream.bind().map_err(candle_core::Error::wrap)?;
    let copy = |dst: u64, src: *const core::ffi::c_void, kind| {
        hip::check(unsafe {
            sys::hipMemcpyAsync(
                dst as *mut core::ffi::c_void,
                src,
                block_size_in_bytes,
                kind,
                stream.raw(),
            )
        })
        .map_err(candle_core::Error::wrap)
    };
    match &*src_storage {
        Storage::Rocm(s) => {
            if s.device.location() != dst_dev.location() {
                candle_core::bail!(
                    "Tensors must be on the same device to copy, got locations {:?} (src) and {:?} (dst).",
                    s.device.location(),
                    dst_dev.location()
                );
            }
            let src_ptr = storage_ptr(&s.slice, src_layout.start_offset())?;
            for (src_block, dst_block) in block_mapping {
                copy(
                    dst_ptr + (dst_block * block_size_in_bytes) as u64,
                    (src_ptr + (src_block * block_size_in_bytes) as u64) as *const _,
                    sys::hipMemcpyDeviceToDevice,
                )?;
            }
        }
        Storage::Cpu(s) => {
            let src_slice = s.as_slice::<u8>()?;
            for (src_block, dst_block) in block_mapping {
                let src_offset = src_block * block_size_in_bytes;
                copy(
                    dst_ptr + (dst_block * block_size_in_bytes) as u64,
                    src_slice[src_offset..src_offset + block_size_in_bytes].as_ptr() as *const _,
                    sys::hipMemcpyHostToDevice,
                )?;
            }
            // The host buffer must stay valid until the copies complete.
            stream.synchronize().map_err(candle_core::Error::wrap)?;
        }
        _ => candle_core::bail!("swap_blocks source must be a CPU or ROCm tensor"),
    }
    Ok(())
}
