//! HIP graph replay of Qwen4-Exp decode steps on ROCm.
//!
//! A decode step launches a few thousand kernels; on one host thread that launch cost, not the
//! GPU, bounds the step. The first step of each batch size runs under stream capture with its
//! per-step inputs (token ids, paged block tables and lengths, slot mappings, positions,
//! recurrent-state slots and the PLE embeddings) copied into buffers the graph owns; later
//! steps of that size refresh those buffers and replay the graph.
//!
//! The decode forward is graph-safe by construction: QSA decode runs in sparse mode with a
//! fixed block budget and reads kv lengths from the device, recurrent state is updated in place
//! through the slot tensor, and the PLE host gather happens before the graph runs.

use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::rocm_backend::hip::HipGraph;
use candle_core::{DType, Device, DeviceLocation, Result, Tensor};

use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{ModelForwardContext, RecurrentBatchKind, RecurrentMetadata};

const ENABLE_ENV: &str = "MISTRALRS_ROCM_DECODE_GRAPHS";
pub(super) const MAX_GRAPH_BATCH: usize = 8;

pub(super) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(ENABLE_ENV).is_ok_and(|v| v != "0"))
}

/// The buffers one captured graph reads its inputs from, and the logits it writes.
struct Entry {
    graph: HipGraph,
    inputs: StaticInputs,
    output: Tensor,
}

struct StaticInputs {
    input_ids: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    slot_mappings: Tensor,
    positions: Tensor,
    state_indices: Tensor,
    ple: Option<Tensor>,
}

/// One step's inputs, as the pipeline produced them.
pub(super) struct StepInputs<'a> {
    pub input_ids: &'a Tensor,
    pub block_tables: &'a Tensor,
    pub context_lens: &'a Tensor,
    pub slot_mappings: &'a Tensor,
    pub positions: &'a Tensor,
    pub state_indices: &'a Tensor,
    pub ple: Option<&'a Tensor>,
}

impl StaticInputs {
    /// Fresh buffers holding `step`; block tables get room for the longest context.
    fn new(step: &StepInputs<'_>, max_blocks: usize) -> Result<Self> {
        let (batch, width) = step.block_tables.dims2()?;
        if width > max_blocks {
            candle_core::bail!("block table width {width} exceeds the graph's {max_blocks}");
        }
        let block_tables = Tensor::zeros(
            (batch, max_blocks),
            step.block_tables.dtype(),
            step.block_tables.device(),
        )?;
        block_tables.slice_set(step.block_tables, 1, 0)?;
        Ok(Self {
            input_ids: step.input_ids.copy()?,
            block_tables,
            context_lens: step.context_lens.copy()?,
            slot_mappings: step.slot_mappings.copy()?,
            positions: step.positions.copy()?,
            state_indices: step.state_indices.copy()?,
            ple: step.ple.map(Tensor::copy).transpose()?,
        })
    }

    /// Copies `step` into the buffers in place.
    fn refresh(&self, step: &StepInputs<'_>) -> Result<()> {
        if step.block_tables.dim(1)? > self.block_tables.dim(1)? {
            candle_core::bail!("block table outgrew the captured decode graph");
        }
        self.input_ids.slice_set(step.input_ids, 0, 0)?;
        self.block_tables.slice_set(step.block_tables, 1, 0)?;
        self.context_lens.slice_set(step.context_lens, 0, 0)?;
        self.slot_mappings.slice_set(step.slot_mappings, 0, 0)?;
        self.positions.slice_set(step.positions, 0, 0)?;
        self.state_indices.slice_set(step.state_indices, 0, 0)?;
        match (&self.ple, step.ple) {
            (Some(dst), Some(src)) => dst.slice_set(src, 0, 0),
            (None, None) => Ok(()),
            _ => candle_core::bail!("PLE embeddings appeared or vanished between decode steps"),
        }
    }

    /// `metadata` with its per-step tensors on `location` replaced by these buffers.
    fn metadata(
        &self,
        metadata: &PagedAttentionInputMetadata,
        location: DeviceLocation,
    ) -> PagedAttentionInputMetadata {
        let mut stable = metadata.clone();
        let only = |t: &Tensor| HashMap::from([(location, t.clone())]);
        stable.block_tables = Some(only(&self.block_tables));
        stable.full_block_tables = Some(only(&self.block_tables));
        stable.context_lens = Some(only(&self.context_lens));
        stable.full_context_lens = Some(only(&self.context_lens));
        stable.slot_mappings = only(&self.slot_mappings);
        stable.rope_positions = Some(only(&self.positions));
        stable
    }
}

/// Captured decode graphs, one per batch size. Batch sizes whose capture failed run eagerly.
#[derive(Default)]
pub(super) struct DecodeGraphs {
    entries: Mutex<HashMap<usize, Entry>>,
    failed: Mutex<std::collections::HashSet<usize>>,
}

/// What the runner needs from the model to capture a step.
pub(super) struct CaptureHooks<'a, F>
where
    F: FnMut(&Tensor, &ModelForwardContext<'_>) -> Result<Tensor>,
{
    pub device: &'a Device,
    pub max_blocks: usize,
    pub set_ple: &'a dyn Fn(Option<Tensor>),
    pub set_state_indices: &'a dyn Fn(Tensor),
    /// Runs the model's forward pass on these token ids against `ctx`.
    pub forward: F,
}

impl DecodeGraphs {
    pub(super) fn has_failed(&self, batch: usize) -> bool {
        self.failed.lock().unwrap().contains(&batch)
    }

    /// Replays (capturing first if needed) the decode graph for this step's batch size and
    /// returns a copy of its logits.
    pub(super) fn run<F>(
        &self,
        step: StepInputs<'_>,
        ctx: &ModelForwardContext<'_>,
        hooks: CaptureHooks<'_, F>,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &ModelForwardContext<'_>) -> Result<Tensor>,
    {
        let batch = step.input_ids.dim(0)?;
        let rocm = hooks.device.as_rocm_device()?;
        let mut entries = self.entries.lock().unwrap();
        if let Some(entry) = entries.get(&batch) {
            let timing = std::env::var_os("MISTRALRS_GRAPH_TIMING").is_some();
            if timing {
                hooks.device.synchronize()?;
            }
            let t0 = std::time::Instant::now();
            entry.inputs.refresh(&step)?;
            rocm.launch_graph(&entry.graph)?;
            let launched = t0.elapsed();
            if timing {
                hooks.device.synchronize()?;
                eprintln!(
                    "GRAPH_TIMING batch={batch} launch_host={:.3}ms gpu={:.3}ms",
                    launched.as_secs_f64() * 1e3,
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            return entry.output.copy();
        }

        let inputs = StaticInputs::new(&step, hooks.max_blocks)?;
        let (kv_caches, metadata) = ctx
            .paged_metadata()
            .ok_or_else(|| candle_core::Error::msg("decode graphs need paged attention"))?;
        let metadata = inputs.metadata(metadata, hooks.device.location());
        let host_slots = ctx
            .recurrent_metadata()
            .and_then(RecurrentMetadata::state_indices_host)
            .map(<[u32]>::to_vec);
        let stable_ctx = ModelForwardContext::new(
            ctx.seqlen_offsets(),
            ctx.context_lens(),
            ctx.position_ids(),
            Some((&kv_caches, &metadata)),
            ctx.flash_params(),
        )
        .with_recurrent_metadata(Some(RecurrentMetadata::new(
            RecurrentBatchKind::Decode,
            inputs.state_indices.clone(),
            host_slots,
        )));
        (hooks.set_state_indices)(inputs.state_indices.clone());
        (hooks.set_ple)(inputs.ple.clone());
        let mut forward = hooks.forward;

        hooks.device.synchronize()?;
        rocm.begin_capture()?;
        let captured = forward(&inputs.input_ids, &stable_ctx);
        let graph = rocm.end_capture();
        let outcome = match (captured, graph) {
            (Ok(output), Ok(graph)) => {
                rocm.launch_graph(&graph)?;
                let logits = output.copy()?;
                entries.insert(
                    batch,
                    Entry {
                        graph,
                        inputs,
                        output,
                    },
                );
                tracing::info!("Captured a ROCm decode graph for batch size {batch}");
                Ok(logits)
            }
            (captured, graph) => {
                let err = captured.err().or(graph.err()).expect("one side failed");
                tracing::warn!(
                    "ROCm decode graph capture failed for batch size {batch}; running eagerly: {err}"
                );
                self.failed.lock().unwrap().insert(batch);
                // Nothing ran during the capture: run this step for real.
                forward(&inputs.input_ids, &stable_ctx)
            }
        };
        (hooks.set_ple)(None);
        outcome
    }
}

/// Whether the step's tensors are the shapes a decode graph takes.
pub(super) fn step_shapes_ok(step: &StepInputs<'_>) -> Result<bool> {
    let batch = step.input_ids.dim(0)?;
    Ok(step.input_ids.dims() == [batch, 1]
        && step.block_tables.dims2()?.0 == batch
        && step.context_lens.elem_count() == batch
        && step.slot_mappings.elem_count() == batch
        && step.positions.elem_count() == batch
        && step.state_indices.elem_count() == batch
        && step.state_indices.dtype() != DType::F32)
}
