//! RCCL, ROCm's NCCL-compatible collectives, for tensor parallelism on AMD GPUs. Each rank is
//! its own process with one GPU; collectives run on that GPU's candle stream, so they stay
//! ordered with the surrounding kernels without host synchronization.

use std::ffi::{c_char, c_int, c_void, CStr};

use candle_core::{DType, Device, Result, Tensor};

#[repr(C)]
#[derive(Clone, Copy)]
struct UniqueId {
    internal: [c_char; 128],
}

type RawComm = *mut c_void;

const NCCL_SUCCESS: c_int = 0;
const NCCL_SUM: c_int = 0;
const NCCL_MAX: c_int = 2;
const NCCL_UINT8: c_int = 1;
const NCCL_UINT32: c_int = 3;
/// Largest reduction (in bf16 elements) the peer all-reduce takes; a decode step's are a few
/// tens of thousands, and larger ones (prefill chunks) go through RCCL.
const PEER_ALLREDUCE_MAX_ELEMS: usize = 1 << 16;
const NCCL_FLOAT16: c_int = 6;
const NCCL_FLOAT32: c_int = 7;
const NCCL_BFLOAT16: c_int = 9;

#[link(name = "rccl")]
extern "C" {
    fn ncclGetUniqueId(id: *mut UniqueId) -> c_int;
    fn ncclCommInitRank(comm: *mut RawComm, nranks: c_int, id: UniqueId, rank: c_int) -> c_int;
    fn ncclCommDestroy(comm: RawComm) -> c_int;
    fn ncclGetErrorString(result: c_int) -> *const c_char;
    fn ncclAllReduce(
        send: *const c_void,
        recv: *mut c_void,
        count: usize,
        dtype: c_int,
        op: c_int,
        comm: RawComm,
        stream: *mut c_void,
    ) -> c_int;
    fn ncclAllGather(
        send: *const c_void,
        recv: *mut c_void,
        send_count: usize,
        dtype: c_int,
        comm: RawComm,
        stream: *mut c_void,
    ) -> c_int;
}

#[link(name = "amdhip64")]
extern "C" {
    fn hipSetDevice(device: c_int) -> c_int;
}

fn check(result: c_int, what: &str) -> Result<()> {
    if result == NCCL_SUCCESS {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(ncclGetErrorString(result)) };
    candle_core::bail!("RCCL {what} failed: {}", msg.to_string_lossy())
}

/// A fresh communicator id, made by rank 0 and handed to the other ranks.
pub(crate) fn new_id() -> [c_char; 128] {
    let mut id = UniqueId { internal: [0; 128] };
    let result = unsafe { ncclGetUniqueId(&mut id) };
    check(result, "ncclGetUniqueId").expect("Failed to create RCCL id");
    id.internal
}

#[derive(Debug)]
pub struct RcclComm {
    comm: RawComm,
    rank: usize,
    world_size: usize,
    device: Device,
    peer: Option<candle_core::rocm_backend::peer::PeerAllReduce>,
}

// RCCL communicators may be used from any thread as long as calls are not concurrent; each rank
// drives its communicator from the model's single forward thread.
unsafe impl Send for RcclComm {}
unsafe impl Sync for RcclComm {}

impl RcclComm {
    pub fn from_device(
        id: &[c_char; 128],
        dev: &Device,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let ordinal = dev.as_rocm_device()?.ordinal();
        tracing::info!(
            "Initializing RCCL communicator: rank={rank}, world_size={world_size}, device={ordinal}"
        );
        let mut comm: RawComm = std::ptr::null_mut();
        unsafe {
            if hipSetDevice(ordinal as c_int) != 0 {
                candle_core::bail!("hipSetDevice({ordinal}) failed before RCCL init");
            }
            check(
                ncclCommInitRank(
                    &mut comm,
                    world_size as c_int,
                    UniqueId { internal: *id },
                    rank as c_int,
                ),
                "ncclCommInitRank",
            )?;
        }
        let mut this = Self {
            comm,
            rank,
            world_size,
            device: dev.clone(),
            peer: None,
        };
        if std::env::var_os("MISTRALRS_NO_PEER_ALLREDUCE").is_none() {
            match this.connect_peers() {
                Ok(peer) => this.peer = Some(peer),
                Err(err) => tracing::warn!("Peer all-reduce unavailable, using RCCL only: {err}"),
            }
        }
        Ok(this)
    }

    /// Shares each rank's peer all-reduce buffer with the others through an RCCL all-gather
    /// of their IPC handles.
    fn connect_peers(&self) -> Result<candle_core::rocm_backend::peer::PeerAllReduce> {
        use candle_core::rocm_backend::peer::{PeerAllReduceSetup, HANDLE_BYTES};
        let (setup, handle) = PeerAllReduceSetup::new(
            self.device.as_rocm_device()?,
            self.rank,
            self.world_size,
            PEER_ALLREDUCE_MAX_ELEMS,
        )?;
        let local = Tensor::from_vec(handle.to_vec(), HANDLE_BYTES, &self.device)?;
        let gathered = unsafe { Tensor::empty(HANDLE_BYTES * self.world_size, DType::U8, &self.device) }?;
        let ptr = |t: &Tensor| -> Result<u64> {
            let (s, l) = t.storage_and_layout();
            let candle_core::Storage::Rocm(s) = &*s else {
                candle_core::bail!("RCCL collectives need rocm tensors");
            };
            Ok(s.as_rocm_slice::<u8>()?.slice(l.start_offset()..).device_ptr())
        };
        unsafe {
            check(
                ncclAllGather(
                    ptr(&local)? as *const c_void,
                    ptr(&gathered)? as *mut c_void,
                    HANDLE_BYTES,
                    NCCL_UINT8,
                    self.comm,
                    stream(&local)?,
                ),
                "ncclAllGather",
            )?;
        }
        let bytes = gathered.to_vec1::<u8>()?;
        let handles = bytes
            .chunks_exact(HANDLE_BYTES)
            .map(|c| c.try_into().expect("handle-sized chunk"))
            .collect::<Vec<[u8; HANDLE_BYTES]>>();
        let peer = setup.connect(&handles)?;
        tracing::info!("Peer all-reduce connected across {} ranks", self.world_size);
        Ok(peer)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

impl Drop for RcclComm {
    fn drop(&mut self) {
        unsafe {
            ncclCommDestroy(self.comm);
        }
    }
}

fn nccl_dtype(dtype: DType) -> Result<c_int> {
    match dtype {
        DType::BF16 => Ok(NCCL_BFLOAT16),
        DType::F16 => Ok(NCCL_FLOAT16),
        DType::F32 => Ok(NCCL_FLOAT32),
        dtype => candle_core::bail!("RCCL: unsupported dtype {dtype:?}"),
    }
}

/// Device address of a contiguous rocm tensor's first element.
fn device_ptr(t: &Tensor) -> Result<u64> {
    use candle_core::rocm_backend::RocmDType;
    fn ptr<T: RocmDType>(s: &candle_core::RocmStorage, offset: usize) -> Result<u64> {
        Ok(s.as_rocm_slice::<T>()?.slice(offset..).device_ptr())
    }
    let (s, l) = t.storage_and_layout();
    let candle_core::Storage::Rocm(s) = &*s else {
        candle_core::bail!("RCCL collectives need rocm tensors");
    };
    match t.dtype() {
        DType::BF16 => ptr::<half::bf16>(s, l.start_offset()),
        DType::F16 => ptr::<half::f16>(s, l.start_offset()),
        DType::F32 => ptr::<f32>(s, l.start_offset()),
        dtype => candle_core::bail!("RCCL: unsupported dtype {dtype:?}"),
    }
}

fn stream(t: &Tensor) -> Result<*mut c_void> {
    Ok(t.device().as_rocm_device()?.hip_stream().raw() as *mut c_void)
}

pub fn sum_all_reduce(comm: &RcclComm, xs: &Tensor) -> Result<Tensor> {
    if let Some(peer) = comm.peer.as_ref().filter(|peer| peer.supports(xs)) {
        return peer.all_reduce(xs);
    }
    let xs = xs.contiguous()?;
    let out = unsafe { Tensor::empty(xs.shape(), xs.dtype(), xs.device()) }?;
    unsafe {
        check(
            ncclAllReduce(
                device_ptr(&xs)? as *const c_void,
                device_ptr(&out)? as *mut c_void,
                xs.elem_count(),
                nccl_dtype(xs.dtype())?,
                NCCL_SUM,
                comm.comm,
                stream(&xs)?,
            ),
            "ncclAllReduce",
        )?;
    }
    Ok(out)
}

/// Concatenates every rank's `xs` along `dim`.
pub fn all_gather(comm: &RcclComm, xs: &Tensor, dim: usize) -> Result<Tensor> {
    let world = comm.world_size;
    // RCCL gathers rank-major, so move `dim` to the front, gather, and move it back.
    let front = if dim == 0 {
        xs.contiguous()?
    } else {
        xs.transpose(0, dim)?.contiguous()?
    };
    let mut dims = front.dims().to_vec();
    dims[0] *= world;
    let out = unsafe { Tensor::empty(dims, front.dtype(), front.device()) }?;
    unsafe {
        check(
            ncclAllGather(
                device_ptr(&front)? as *const c_void,
                device_ptr(&out)? as *mut c_void,
                front.elem_count(),
                nccl_dtype(front.dtype())?,
                comm.comm,
                stream(&front)?,
            ),
            "ncclAllGather",
        )?;
    }
    if dim == 0 {
        Ok(out)
    } else {
        out.transpose(0, dim)?.contiguous()
    }
}

/// The largest `value` across ranks, read back to the host. Ranks call this at the same points
/// of their host loops to agree on decisions that must not diverge.
pub fn all_reduce_max_u32(comm: &RcclComm, value: u32) -> Result<u32> {
    let xs = Tensor::new(&[value], &comm.device)?;
    let out = unsafe { Tensor::empty(1, DType::U32, &comm.device) }?;
    let ptr = |t: &Tensor| -> Result<u64> {
        let (s, l) = t.storage_and_layout();
        let candle_core::Storage::Rocm(s) = &*s else {
            candle_core::bail!("RCCL collectives need rocm tensors");
        };
        Ok(s.as_rocm_slice::<u32>()?.slice(l.start_offset()..).device_ptr())
    };
    unsafe {
        check(
            ncclAllReduce(
                ptr(&xs)? as *const c_void,
                ptr(&out)? as *mut c_void,
                1,
                NCCL_UINT32,
                NCCL_MAX,
                comm.comm,
                stream(&xs)?,
            ),
            "ncclAllReduce",
        )?;
    }
    out.to_vec1::<u32>().map(|v| v[0])
}
