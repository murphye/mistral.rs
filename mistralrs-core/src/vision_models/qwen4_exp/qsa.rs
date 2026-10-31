#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use super::config::{QsaConfig, TextConfig};
use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{GemmaRmsNorm, Qwen3VLRotaryEmbedding, Sdpa},
    paged_attention::{load_fp8_attention_scales, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        KvCache,
    },
};

/// Per-token aux cache row: `[indexer key | cos | sin]`.
pub(super) fn aux_dim(qsa: &QsaConfig, half_rot: usize) -> usize {
    qsa.head_dim + 2 * half_rot
}

/// How a forward runs its QSA layers, decided once per step on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QsaMode {
    /// Every query sees at most budget + ratio - 1 tokens: the selection keeps them all.
    Dense,
    /// Block-sparse attention; `max_blocks` bounds the complete blocks of any query.
    Sparse { max_blocks: usize },
}

impl QsaMode {
    pub(super) fn for_max_kv_len(qsa: &QsaConfig, max_kv_len: usize) -> Self {
        if max_kv_len <= qsa.max_selected_tokens() {
            Self::Dense
        } else {
            Self::Sparse {
                max_blocks: max_kv_len / qsa.compress_ratio,
            }
        }
    }
}

/// Per-forward state shared by every QSA layer.
pub(super) struct QsaStep<'a> {
    pub mode: QsaMode,
    /// Absolute kv position of every flattened token, host side (non-paged path only).
    pub positions: &'a [Option<usize>],
#[cfg(any(feature = "cuda", feature = "rocm"))]
    pub layout: Option<&'a super::kernels::TokenLayout>,
    /// Each sequence's physical recurrent state slot (u32, batch order), the logical slot
    /// capacity and the checkpoint lanes per logical slot: the paged path keys its raw
    /// indexer-key carry by logical slot (`physical / lanes`).
    pub state_slots: Option<QsaStateSlots>,
}

#[derive(Clone)]
pub(super) struct QsaStateSlots {
    pub slots: Tensor,
    pub capacity: usize,
    pub lanes: usize,
}

/// Bytes of one raw indexer-key row: int8 per element plus an f32 scale (the kernels'
/// Q4_QSA_RAW_ROW_BYTES).
const QSA_RAW_ROW_BYTES: usize = 4;

#[derive(Clone)]
#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
struct QsaAuxCache {
    /// Raw indexer keys of each sequence's open compressed block and the one before it,
    /// `[slot_capacity * 2 * ratio, head_dim + 4]` u8, indexed by recurrent state slot. Keys
    /// of completed blocks are not kept: only their pooled block key is.
    carry: Tensor,
    /// cos then sin of each compressed block's first position, `[slots / ratio, 2 * half_rot]`:
    /// the finalize kernel rotates the pooled block key by the block's first position only.
    block_cs: Tensor,
    /// Pooled, normed and rotated key per completed block, `[slots / ratio, head_dim]`.
    block_keys: Tensor,
}

pub(super) struct QsaAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    pub(super) q_norm: GemmaRmsNorm,
    pub(super) k_norm: GemmaRmsNorm,
    index_qk_proj: Arc<dyn QuantMethod>,
    pub(super) index_q_norm: GemmaRmsNorm,
    pub(super) index_k_norm: GemmaRmsNorm,
    // (1 + w) in f32 for the fused block finalize
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    index_k_norm_f32: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    half_rot: usize,
    qsa: QsaConfig,
    pub(super) rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    eps: f64,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    aux_cache: Mutex<Option<QsaAuxCache>>,
}

impl QsaAttention {
    /// `vb_sa` and `vb_norms` are the `self_attn` builders already placed on the layer device, the
    /// former with ISQ applied.
    pub(super) fn load(
        vb_sa: ShardedVarBuilder,
        vb_norms: ShardedVarBuilder,
        cfg: &TextConfig,
        qsa: QsaConfig,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        use_paged_attention: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        // Tensor parallelism splits the attention heads; the indexer stays replicated so every
        // rank selects the same blocks without another collective.
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let n_kv_groups = mistralrs_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm)?;
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("q_proj"),
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("v_proj"),
        )?;
        let num_heads = num_heads / comm.world_size();
        let num_kv_heads = num_heads / n_kv_groups;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("o_proj"),
        )?;
        let index_qk_proj = ReplicatedLayer::new(
            cfg.hidden_size,
            (qsa.n_heads + 1) * qsa.head_dim,
            &cfg.quantization_config,
            false,
            vb_sa.pp("indexer").pp("index_qk_proj"),
        )?;
        #[cfg(feature = "cutile")]
        if use_paged_attention {
            mistralrs_quant::cutile::register_qsa_shape(mistralrs_quant::cutile::QsaWarmShape {
                ratio: qsa.compress_ratio,
                topk: qsa.block_topk(),
                n_q_heads: num_heads,
                n_kv_heads: num_kv_heads,
                block_size: crate::paged_attention::DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
            });
        }
        let paged_attn = if use_paged_attention {
            Some(PagedAttention::new_with_fp8_attention_scales(
                head_dim,
                vb_norms.device(),
                None,
                load_fp8_attention_scales(&vb_norms)?,
            )?)
        } else {
            None
        };
        let index_k_norm = GemmaRmsNorm::new(
            qsa.head_dim,
            cfg.rms_norm_eps,
            vb_norms.pp("indexer").pp("k_layernorm"),
        )?;
        let index_k_norm_f32 = (index_k_norm.original_weight().to_dtype(DType::F32)? + 1.0)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_norms.pp("q_norm"))?,
            k_norm: GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_norms.pp("k_norm"))?,
            index_qk_proj,
            index_q_norm: GemmaRmsNorm::new(
                qsa.head_dim,
                cfg.rms_norm_eps,
                vb_norms.pp("indexer").pp("q_layernorm"),
            )?,
            index_k_norm,
            index_k_norm_f32,
            num_heads,
            num_kv_heads,
            head_dim,
            half_rot: cfg.rot_dim() / 2,
            qsa,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            eps: cfg.rms_norm_eps,
            aux_cache: Mutex::new(None),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        step: &QsaStep<'_>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        // ROCm decode: the three GGUF attention projections of `x` in one launch (the indexer
        // projection is an unquantized linear in these checkpoints).
        #[cfg(feature = "rocm")]
        let fused = mistralrs_quant::try_forward_multi_gguf_rocm(
            x,
            &[&*self.q_proj, &*self.k_proj, &*self.v_proj],
        )?
        .map(|outputs| {
            let [q_gate, k, v]: [Tensor; 3] =
                outputs.try_into().expect("three projections requested");
            (q_gate, k, v)
        });
        #[cfg(not(feature = "rocm"))]
        let fused: Option<(Tensor, Tensor, Tensor)> = None;
        let (q_gate, k, v) = match fused {
            Some(fused) => fused,
            None => crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?,
        };
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        // The gate half of the contiguous `[tokens, heads, 2 * head_dim]` projection, for the
        // attention kernel to fold in.
        let gate_view = q_gate.is_contiguous().then_some((
            &q_gate,
            self.head_dim,
            self.num_heads * self.head_dim * 2,
            self.head_dim * 2,
        ));
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let cos_sin = (
            cos_sin.0.to_device(x.device())?,
            cos_sin.1.to_device(x.device())?,
        );
        // Token-major output buffers: the cache write and the attention kernel consume q and k
        // as `[tokens, heads, d]`, so their later `transpose(1, 2).contiguous()` copies nothing.
        let (q, k) = self.rotary_emb.forward_qk_norm_layout(
            &cos_sin,
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            true,
        )?;

        let index_qk = self.index_qk_proj.forward(x)?;
        let index_q_dim = self.qsa.n_heads * self.qsa.head_dim;
        // ROCm decode: the indexer query's norm and rope and the raw key copy in one launch.
        #[cfg(feature = "rocm")]
        let fused_index = if x.device().is_rocm()
            && self.paged_attn.is_some()
            && b_sz * seq_len < 64
            && cos_sin.0.dtype() == index_qk.dtype()
            && self.index_q_norm.weight().dtype() == index_qk.dtype()
        {
            let n_tokens = b_sz * seq_len;
            let (index_q, raw_key) = super::kernels::index_q_prep(
                &index_qk.reshape((n_tokens, ()))?,
                self.index_q_norm.weight(),
                self.index_q_norm.eps(),
                &cos_sin.0.reshape((n_tokens, ()))?,
                &cos_sin.1.reshape((n_tokens, ()))?,
                self.qsa.n_heads,
                self.qsa.head_dim,
            )?;
            Some((
                index_q.reshape((b_sz, seq_len, self.qsa.n_heads, self.qsa.head_dim))?,
                raw_key.reshape((b_sz, seq_len, self.qsa.head_dim))?,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "rocm"))]
        let fused_index: Option<(Tensor, Tensor)> = None;
        let (index_q, raw_key) = match fused_index {
            Some(fused) => fused,
            None => {
                let index_q = index_qk.narrow(D::Minus1, 0, index_q_dim)?.reshape((
                    b_sz,
                    seq_len,
                    self.qsa.n_heads,
                    self.qsa.head_dim,
                ))?;
                // Unpaged dense steps attend to every key and never read the indexer query.
                let index_q =
                    if self.paged_attn.is_some() || matches!(step.mode, QsaMode::Sparse { .. }) {
                        self.index_query(&index_q, &cos_sin)?
                    } else {
                        index_q
                    };
                let raw_key = index_qk.narrow(D::Minus1, index_q_dim, self.qsa.head_dim)?;
                (index_q, raw_key)
            }
        };

        let (y, gated) = match (&self.paged_attn, metadata) {
            (Some(paged_attn), Some(((key_cache, value_cache), input_metadata))) => self
                .forward_paged(PagedQsaInputs {
                    paged_attn,
                    q: &q,
                    k: &k,
                    v: &v,
                    gate: gate_view,
                    index_q: &index_q,
                    raw_key: &raw_key,
                    cos_sin: &cos_sin,
                    attention_mask,
                    key_cache,
                    value_cache,
                    metadata: input_metadata,
                    flash_params,
                    step,
                })?,
            (Some(paged_attn), None) => {
                let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                let y = paged_attn.forward(
                    &q,
                    &k,
                    &v.transpose(1, 2)?.contiguous()?,
                    attention_mask,
                    None,
                    None,
                    &input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?;
                (y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?, false)
            }
            (None, _) => (self.forward_unpaged(UnpagedQsaInputs {
                q: &q,
                k: &k,
                v: &v,
                index_q: &index_q,
                raw_key: &raw_key,
                cos_sin: &cos_sin,
                attention_mask,
                kv_cache: kv_cache.ok_or_else(|| {
                    candle_core::Error::msg("unpaged Qwen4-Exp QSA requires a KV cache")
                })?,
                flash_params,
                step,
            })?, false),
        };

        if gated {
            return self.o_proj.forward(&y);
        }
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        self.o_proj.forward(&y.broadcast_mul(&gate)?)
    }

    /// Normalized, rotated indexer queries `[b, s, heads, head_dim]`.
    fn index_query(&self, index_q: &Tensor, (cos, sin): &(Tensor, Tensor)) -> Result<Tensor> {
        let q = candle_nn::ops::rms_norm(
            &index_q.contiguous()?,
            self.index_q_norm.weight(),
            self.eps as f32,
        )?;
        rope_partial_neox(&q, cos, sin)
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn aux_cache(
        &self,
        key_cache: &Tensor,
        block_size: usize,
        dtype: DType,
        slot_capacity: usize,
    ) -> Result<QsaAuxCache> {
        if !block_size.is_multiple_of(self.qsa.compress_ratio) {
            candle_core::bail!(
                "Qwen4-Exp QSA paged block size {block_size} must be divisible by compression ratio {}",
                self.qsa.compress_ratio
            );
        }
        let mut aux = self.aux_cache.lock().expect("QSA aux cache poisoned");
        let num_blocks = key_cache.dim(0)?;
        let ratio = self.qsa.compress_ratio;
        let carry_rows = slot_capacity * 2 * ratio;
        let row_bytes = self.qsa.head_dim + QSA_RAW_ROW_BYTES;
        if let Some(existing) = aux.as_mut() {
            if existing.block_keys.dim(0)? == num_blocks * block_size / ratio
                && existing.block_keys.device().same_device(key_cache.device())
            {
                // The slot pool can grow; keep the carried keys of the slots that exist.
                if existing.carry.dim(0)? < carry_rows {
                    let grown =
                        Tensor::zeros((carry_rows, row_bytes), DType::U8, key_cache.device())?;
                    grown.slice_set(&existing.carry, 0, 0)?;
                    existing.carry = grown;
                }
                return Ok(existing.clone());
            }
        }
        let half_rot = self.rotary_emb_half_dim();
        let cache = QsaAuxCache {
            carry: Tensor::zeros((carry_rows, row_bytes), DType::U8, key_cache.device())?,
            block_cs: Tensor::zeros(
                (num_blocks * block_size / self.qsa.compress_ratio, 2 * half_rot),
                dtype,
                key_cache.device(),
            )?,
            block_keys: Tensor::zeros(
                (
                    num_blocks * block_size / self.qsa.compress_ratio,
                    self.qsa.head_dim,
                ),
                dtype,
                key_cache.device(),
            )?,
        };
        *aux = Some(cache.clone());
        Ok(cache)
    }

    fn rotary_emb_half_dim(&self) -> usize {
        self.half_rot
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    /// The attention output and whether the output gate was already applied.
    fn forward_paged(&self, inputs: PagedQsaInputs<'_>) -> Result<(Tensor, bool)> {
        let PagedQsaInputs {
            paged_attn,
            q,
            k,
            v,
            gate,
            index_q,
            raw_key,
            cos_sin,
            attention_mask,
            key_cache,
            value_cache,
            metadata,
            flash_params,
            step,
        } = inputs;
        let (b_sz, _, seq_len, _) = q.dims4()?;
#[cfg(any(feature = "cuda", feature = "rocm"))]
        if super::kernel_device(q.device()) {
            use super::kernels;
            let block_size = metadata
                .block_size
                .ok_or_else(|| candle_core::Error::msg("QSA requires a paged block size"))?;
            let location = q.device().location();
            let block_tables = metadata
                .block_tables
                .as_ref()
                .and_then(|tables| tables.get(&location))
                .ok_or_else(|| candle_core::Error::msg("QSA requires paged block tables"))?;
            let layout = step
                .layout
                .ok_or_else(|| candle_core::Error::msg("QSA requires a token layout"))?;
            // Multi-token decode metadata carries one block table per query row; QSA wants one per sequence
            let block_tables = &per_sequence_block_tables(block_tables, layout.n_seqs)?;
            // Prompt metadata carries per-token positions in `context_lens`; only decode has lengths
            // Decode-row lengths are per query row; the last row of each sequence holds its kv length
            let kv_lens = &match &layout.kv_lens {
                Some(kv_lens) => kv_lens.clone(),
                None => last_row_per_sequence(
                    metadata
                        .context_lens
                        .as_ref()
                        .and_then(|lens| lens.get(&location))
                        .ok_or_else(|| {
                            candle_core::Error::msg("QSA requires paged context lengths")
                        })?,
                    layout.n_seqs,
                )?,
            };
            let slot_mapping = metadata
                .slot_mappings
                .get(&location)
                .ok_or_else(|| candle_core::Error::msg("QSA requires paged slot mappings"))?;
            // U8 is the 4-bit row format; the kernels check the K/V pair in detail.
            if key_cache.dtype() != q.dtype()
                && !matches!(key_cache.dtype(), DType::F8E4M3 | DType::U8)
            {
                candle_core::bail!(
                    "Qwen4-Exp QSA needs a {:?}, f8e4m3 or 4-bit KV cache, got {:?}",
                    q.dtype(),
                    key_cache.dtype()
                );
            }
            if !matches!(block_tables.dtype(), DType::U32 | DType::I32)
                || !matches!(kv_lens.dtype(), DType::U32 | DType::I32)
            {
                candle_core::bail!("QSA expects 32-bit paged block tables and context lengths");
            }
            let paged = kernels::PagedView {
                block_tables,
                kv_lens,
                block_size,
            };
            let shape = kernels::QsaShape {
                ratio: self.qsa.compress_ratio,
                topk: self.qsa.block_topk(),
                half_rot: self.rotary_emb_half_dim(),
                n_index_heads: self.qsa.n_heads,
            };
            let n_tokens = b_sz * seq_len;
            let Some(state_slots) = step.state_slots.as_ref() else {
                candle_core::bail!("Qwen4-Exp paged QSA needs the sequences' recurrent state slots");
            };
            let aux = self.aux_cache(&key_cache, block_size, q.dtype(), state_slots.capacity)?;
            // This step's raw indexer keys, consumed by the finalize and carry update below.
            let scratch = unsafe {
                Tensor::empty(
                    (n_tokens, self.qsa.head_dim + QSA_RAW_ROW_BYTES),
                    DType::U8,
                    q.device(),
                )
            }?;
            kernels::qsa_aux_write(
                &raw_key.reshape((n_tokens, self.qsa.head_dim))?,
                &cos_sin.0.reshape((n_tokens, shape.half_rot))?,
                &cos_sin.1.reshape((n_tokens, shape.half_rot))?,
                slot_mapping,
                &scratch,
                &aux.block_cs,
                shape.ratio,
            )?;
            kernels::qsa_finalize(kernels::QsaFinalizeArgs {
                aux: &scratch,
                carry: &aux.carry,
                state_slots: &state_slots.slots,
                lanes: state_slots.lanes,
                block_cs: &aux.block_cs,
                block_keys: &aux.block_keys,
                layout,
                paged: &paged,
                norm_weight: &self.index_k_norm_f32,
                shape: &shape,
                eps: self.eps,
            })?;
            // On ROCm the token-major (HND) cache is only read by the QSA kernels, so dense mode
            // runs them with every block selected wherever the layer's standard path would read
            // the cache (decode, chunked or prefix-cached prompts). A first prompt chunk without
            // cached context attends over its own bf16 K/V and keeps the exact dense path.
            let hnd_cache = key_cache.rank() == 4;
            let rocm_hnd = cfg!(all(feature = "rocm", not(feature = "cuda"))) && hnd_cache;
            let exact_dense_prompt = metadata.is_first_prompt_chunk
                && metadata
                    .num_cached_tokens
                    .as_ref()
                    .is_none_or(|cached| cached.iter().all(|&n| n == 0))
                && !metadata.has_noncausal_mm_context;
            if std::env::var_os("MISTRALRS_QSA_ROUTE_DEBUG").is_some() {
                eprintln!(
                    "qsa route: mode={:?} hnd={hnd_cache} first_chunk={} cached={:?} noncausal_mm={} exact={exact_dense_prompt} tokens={n_tokens}",
                    step.mode,
                    metadata.is_first_prompt_chunk,
                    metadata.num_cached_tokens,
                    metadata.has_noncausal_mm_context
                );
            }
            let max_blocks = match step.mode {
                QsaMode::Sparse { max_blocks } => Some(max_blocks),
                QsaMode::Dense if rocm_hnd && exact_dense_prompt => {
                    // Written through the HND kernel; attention over this chunk's own bf16 K/V
                    // (the layer's standard prompt path would gather from the cache in the
                    // vLLM layout as soon as the scheduler reports cached tokens, even zero).
                    let mut key_cache = key_cache.clone();
                    let mut value_cache = value_cache.clone();
                    paged_attn.write_cache(
                        &k.transpose(1, 2)?.contiguous()?,
                        &v.contiguous()?,
                        &mut key_cache,
                        &mut value_cache,
                        metadata,
                    )?;
                    let y = Sdpa.run_attention(
                        q,
                        k,
                        &v.transpose(1, 2)?.contiguous()?,
                        attention_mask,
                        Some(flash_params),
                        &self.sdpa_params,
                    )?;
                    return Ok((y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?, false));
                }
                QsaMode::Dense if rocm_hnd => None,
                QsaMode::Dense => {
                    // The dense prompt paths multiply `q` as `[b, heads, tokens, d]`, which the
                    // token-major buffer only is after a copy (the ROCm matmul takes no other
                    // row stride).
                    let y = paged_attn.forward(
                    &q.contiguous()?,
                    k,
                    &v.transpose(1, 2)?.contiguous()?,
                    attention_mask,
                    Some(key_cache.clone()),
                    Some(value_cache.clone()),
                    metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?;
                    return Ok((y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?, false));
                }
            };
            let mut key_cache = key_cache.clone();
            let mut value_cache = value_cache.clone();
            paged_attn.write_cache(
                &k.transpose(1, 2)?.contiguous()?,
                &v.contiguous()?,
                &mut key_cache,
                &mut value_cache,
                metadata,
            )?;
            let index_q = index_q.reshape((n_tokens, self.qsa.n_heads, self.qsa.head_dim))?;
            let (selected, n_selected) = match max_blocks {
                #[cfg(all(feature = "rocm", not(feature = "cuda")))]
                None => kernels::qsa_select_all(layout, &paged, &shape, q.device())?,
                #[cfg(not(all(feature = "rocm", not(feature = "cuda"))))]
                None => unreachable!("dense QSA selects every block only on ROCm HND caches"),
                Some(max_blocks) => match kernels::qsa_select_cutile(
                    &index_q,
                    &aux.block_keys,
                    layout,
                    &paged,
                    &shape,
                    max_blocks,
                )? {
                    Some(selection) => selection,
                    None => kernels::qsa_select(
                        &index_q,
                        &aux.block_keys,
                        layout,
                        &paged,
                        &shape,
                        max_blocks,
                    )?,
                },
            };
            let q_tokens = q.transpose(1, 2)?.contiguous()?.reshape((
                n_tokens,
                self.num_heads,
                self.head_dim,
            ))?;
            let attn_args = kernels::QsaAttentionArgs {
                q: &q_tokens,
                key_cache: &key_cache,
                value_cache: &value_cache,
                layout,
                paged: &paged,
                selected: &selected,
                n_selected: &n_selected,
                shape: &shape,
                n_kv_heads: self.num_kv_heads,
                scale: self.sdpa_params.softmax_scale,
                gate,
                // Dequantization scales of an fp8 cache (unit for bf16/f16 caches).
                kv_scales: {
                    let scales = paged_attn.fp8_attention_scales();
                    (scales.k, scales.v)
                },
            };
            let (y, gated) = match kernels::qsa_attention_cutile(&attn_args)? {
                Some(y) => (y, false),
                None => kernels::qsa_attention(attn_args)?,
            };
            return Ok((y.reshape((b_sz, seq_len, ()))?, gated));
        }
        candle_core::bail!(
            "Qwen4-Exp paged attention requires CUDA; run with paged attention disabled on this device"
        )
    }

    fn forward_unpaged(&self, inputs: UnpagedQsaInputs<'_>) -> Result<Tensor> {
        let UnpagedQsaInputs {
            q,
            k,
            v,
            index_q,
            raw_key,
            cos_sin,
            attention_mask,
            kv_cache,
            flash_params,
            step,
        } = inputs;
        let (b_sz, _, seq_len, _) = q.dims4()?;
        let half_rot = self.rotary_emb_half_dim();
        let aux_width = aux_dim(&self.qsa, half_rot);
        // The indexer state rides along as an extra key head so it batches and truncates with the cache
        let k_all = self.pack_keys(k, raw_key, cos_sin, aux_width)?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (cache_k, cache_v) = kv_cache.append(&k_all, &v)?;
        let keys = cache_k.narrow(1, 0, self.num_kv_heads)?.contiguous()?;

        let y = match step.mode {
            QsaMode::Dense => Sdpa.run_attention(
                q,
                &keys,
                &cache_v,
                attention_mask,
                Some(flash_params),
                &self.sdpa_params,
            )?,
            QsaMode::Sparse { .. } => {
                let aux = cache_k
                    .i((.., self.num_kv_heads, .., ..aux_width))?
                    .contiguous()?;
                let mask = self.selection_mask(index_q, &aux, step, b_sz, seq_len, q.dtype())?;
                Sdpa.run_attention_noflash(
                    q,
                    &keys,
                    &cache_v,
                    Some(&mask),
                    &self.sdpa_params,
                    false,
                )?
            }
        };
        y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))
    }

    /// `k` with the aux head `[raw_key, cos, sin, 0..]` appended along the head axis.
    fn pack_keys(
        &self,
        k: &Tensor,
        raw_key: &Tensor,
        cos_sin: &(Tensor, Tensor),
        aux_width: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "rocm")]
        if k.device().is_rocm() {
            return crate::rocm::moe_glue::qsa_pack_keys(k, raw_key, &cos_sin.0, &cos_sin.1);
        }
        let aux = Tensor::cat(
            &[
                raw_key.contiguous()?,
                cos_sin.0.to_dtype(k.dtype())?,
                cos_sin.1.to_dtype(k.dtype())?,
            ],
            D::Minus1,
        )?
        .pad_with_zeros(D::Minus1, 0, self.head_dim - aux_width)?
        .unsqueeze(1)?;
        Tensor::cat(&[k.clone(), aux], 1)?.contiguous()
    }

    /// Additive `[b, 1, s, kv]` mask admitting each query's selected blocks and incomplete tail.
    fn selection_mask(
        &self,
        index_q: &Tensor,
        aux: &Tensor,
        step: &QsaStep<'_>,
        b_sz: usize,
        seq_len: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let ratio = self.qsa.compress_ratio;
        let topk = self.qsa.block_topk();
        let half_rot = self.rotary_emb_half_dim();
        let kv_len = aux.dim(1)?;
        let n_blocks = kv_len / ratio;
        let d = self.qsa.head_dim;
        let block_starts = aux.narrow(1, 0, n_blocks * ratio)?;
        let pooled = block_starts
            .narrow(D::Minus1, 0, d)?
            .contiguous()?
            .reshape((b_sz, n_blocks, ratio, d))?
            .to_dtype(DType::F32)?
            .mean(2)?
            .to_dtype(aux.dtype())?;
        let pooled = candle_nn::ops::rms_norm(
            &pooled.contiguous()?,
            self.index_k_norm.weight(),
            self.eps as f32,
        )?;
        let starts = block_starts
            .reshape((b_sz, n_blocks, ratio, aux.dim(D::Minus1)?))?
            .i((.., .., 0, ..))?;
        let cos = starts.narrow(D::Minus1, d, half_rot)?.contiguous()?;
        let sin = starts
            .narrow(D::Minus1, d + half_rot, half_rot)?
            .contiguous()?;
        let block_keys = rope_partial_neox(&pooled.unsqueeze(2)?, &cos, &sin)?.squeeze(2)?;
        // [b, s, heads, d] x [b, blocks, d] -> [b, s, heads, blocks]
        let scores = index_q
            .to_dtype(DType::F32)?
            .reshape((b_sz, seq_len * self.qsa.n_heads, d))?
            .matmul(&block_keys.to_dtype(DType::F32)?.transpose(1, 2)?)?
            .relu()?
            .reshape((b_sz, seq_len, self.qsa.n_heads, n_blocks))?
            .sum(2)?;
        let scores = (scores / (d as f64).sqrt())?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        let mut mask = vec![f32::NEG_INFINITY; b_sz * seq_len * kv_len];
        for b in 0..b_sz {
            for i in 0..seq_len {
                let Some(pos) = step.positions[b * seq_len + i] else {
                    continue;
                };
                let row = &mut mask[(b * seq_len + i) * kv_len..(b * seq_len + i + 1) * kv_len];
                let nb = (pos + 1) / ratio;
                let mut blocks = (0..nb).collect::<Vec<_>>();
                if nb > topk {
                    let s = &scores[b][i];
                    blocks.select_nth_unstable_by(topk, |x, y| s[*y].total_cmp(&s[*x]));
                    blocks.truncate(topk);
                }
                for block in blocks {
                    row[block * ratio..(block + 1) * ratio].fill(0.0);
                }
                row[nb * ratio..=pos].fill(0.0);
            }
        }
        Tensor::from_vec(mask, (b_sz, 1, seq_len, kv_len), index_q.device())?.to_dtype(dtype)
    }
}

struct PagedQsaInputs<'a> {
    paged_attn: &'a PagedAttention,
    q: &'a Tensor,
    k: &'a Tensor,
    v: &'a Tensor,
    /// The output gate logits as `(tensor, element offset, token stride, head stride)`, for
    /// the attention kernel to apply (sigmoid included) when it can.
    gate: Option<(&'a Tensor, usize, usize, usize)>,
    index_q: &'a Tensor,
    raw_key: &'a Tensor,
    cos_sin: &'a (Tensor, Tensor),
    attention_mask: &'a AttentionMask,
    key_cache: Tensor,
    value_cache: Tensor,
    metadata: &'a PagedAttentionInputMetadata,
    flash_params: &'a FlashParams,
    step: &'a QsaStep<'a>,
}

struct UnpagedQsaInputs<'a> {
    q: &'a Tensor,
    k: &'a Tensor,
    v: &'a Tensor,
    index_q: &'a Tensor,
    raw_key: &'a Tensor,
    cos_sin: &'a (Tensor, Tensor),
    attention_mask: &'a AttentionMask,
    kv_cache: &'a mut KvCache,
    flash_params: &'a FlashParams,
    step: &'a QsaStep<'a>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn per_sequence_block_tables(block_tables: &Tensor, n_seqs: usize) -> Result<Tensor> {
    let (rows, width) = block_tables.dims2()?;
    if rows == n_seqs {
        return Ok(block_tables.clone());
    }
    if n_seqs == 0 || !rows.is_multiple_of(n_seqs) {
        candle_core::bail!("QSA block tables have {rows} rows for {n_seqs} sequences");
    }
    block_tables
        .reshape((n_seqs, rows / n_seqs, width))?
        .i((.., 0, ..))?
        .contiguous()
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn last_row_per_sequence(lens: &Tensor, n_seqs: usize) -> Result<Tensor> {
    let rows = lens.dim(0)?;
    if rows == n_seqs {
        return Ok(lens.clone());
    }
    if n_seqs == 0 || !rows.is_multiple_of(n_seqs) {
        candle_core::bail!("QSA context lengths have {rows} rows for {n_seqs} sequences");
    }
    let q_len = rows / n_seqs;
    lens.reshape((n_seqs, q_len))?
        .i((.., q_len - 1))?
        .contiguous()
}

/// NeoX RoPE over the first `2 * half` dims of `x: [b, s, heads, d]`; `cos`/`sin` are `[b, s, half]`.
pub(super) fn rope_partial_neox(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "rocm")]
    if x.device().is_rocm() {
        return crate::rocm::moe_glue::rope_partial_neox(x, cos, sin);
    }
    let half = cos.dim(D::Minus1)?;
    let d = x.dim(D::Minus1)?;
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(2)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(2)?;
    let xf = x.to_dtype(DType::F32)?;
    let x1 = xf.narrow(D::Minus1, 0, half)?;
    let x2 = xf.narrow(D::Minus1, half, half)?;
    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    let mut parts = vec![r1, r2];
    if d > 2 * half {
        parts.push(xf.narrow(D::Minus1, 2 * half, d - 2 * half)?);
    }
    Tensor::cat(&parts, D::Minus1)?.to_dtype(x.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_rope_rotates_leading_dims_only() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::arange(0f32, 8., &device)?.reshape((1, 1, 1, 8))?;
        let cos = Tensor::new(&[[[0f32, 1.0]]], &device)?;
        let sin = Tensor::new(&[[[1f32, 0.0]]], &device)?;
        let out = rope_partial_neox(&x, &cos, &sin)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // pairs (0, 2) rotate by 90 degrees, (1, 3) stay, dims 4.. pass through
        assert_eq!(out, [-2.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        Ok(())
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_qsa_pack_keys_matches_cat() -> Result<()> {
        let rocm = Device::new_rocm(0)?;
        let (b, heads, s, d, d_idx, half) = (2, 3, 5, 256, 128, 32);
        let k = Tensor::randn(0f32, 1.0, (b, heads, s, d), &Device::Cpu)?.to_dtype(DType::BF16)?;
        // raw_key as a strided narrow of a wider projection, like index_qk.
        let wide = Tensor::randn(0f32, 1.0, (b, s, 3 * d_idx), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let cos = Tensor::randn(0f32, 1.0, (b, s, half), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let sin = Tensor::randn(0f32, 1.0, (b, s, half), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let aux = Tensor::cat(&[wide.narrow(D::Minus1, 2 * d_idx, d_idx)?, cos.clone(), sin.clone()], D::Minus1)?
            .pad_with_zeros(D::Minus1, 0, d - d_idx - 2 * half)?
            .unsqueeze(1)?;
        let expected = Tensor::cat(&[k.clone(), aux], 1)?.to_dtype(DType::F32)?;
        let got = crate::rocm::moe_glue::qsa_pack_keys(
            &k.to_device(&rocm)?,
            &wide.to_device(&rocm)?.narrow(D::Minus1, 2 * d_idx, d_idx)?,
            &cos.to_device(&rocm)?,
            &sin.to_device(&rocm)?,
        )?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?;
        let diff = (got - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);
        Ok(())
    }

    /// The 4-bit token-major KV cache: rows decode to the written values within half a
    /// quantization step, and attention over fp8, fp8-key/4-bit-value and 4-bit caches stays
    /// close to the bf16 cache through both the prefill (WMMA) and decode kernels.
    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_4bit_kv_cache_tracks_bf16() -> Result<()> {
        use crate::rocm::qwen4_exp as k;
        let rocm = Device::new_rocm(0)?;
        let (tokens, q_heads, kv_heads, d, block_size, ratio) = (96usize, 8, 1, 256, 32, 4);
        let blocks = tokens / block_size;
        let rand = |shape: (usize, usize, usize)| -> Result<Tensor> {
            Tensor::randn(0f32, 1.0, shape, &Device::Cpu)?
                .to_dtype(DType::BF16)?
                .to_device(&rocm)
        };
        let (kx, vx, q) = (
            rand((tokens, kv_heads, d))?,
            rand((tokens, kv_heads, d))?,
            rand((tokens, q_heads, d))?,
        );
        let slots = Tensor::arange(0i64, tokens as i64, &rocm)?;
        let tables = Tensor::new(&[[0u32, 1, 2]], &rocm)?;
        let kv_lens = Tensor::new(&[tokens as u32], &rocm)?;
        let paged = k::PagedView {
            block_tables: &tables,
            kv_lens: &kv_lens,
            block_size,
        };
        let shape = k::QsaShape {
            ratio,
            topk: tokens / ratio,
            half_rot: 32,
            n_index_heads: 1,
        };
        let i4_row = k::ATTN_I4_ROW_BYTES;
        let cache = |dtype: DType, last: usize| {
            Tensor::zeros((blocks, kv_heads, block_size, last), dtype, &rocm)
        };

        // Every element of a 4-bit row decodes within half a step of what was written.
        let (kc, vc) = (cache(DType::U8, i4_row)?, cache(DType::U8, i4_row)?);
        k::kv_write_hnd(&kx, &vx, &kc, &vc, &slots, None, None)?;
        let bytes = kc.flatten_all()?.to_vec1::<u8>()?;
        let want = kx.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for t in 0..tokens {
            let row = &bytes[t * i4_row..(t + 1) * i4_row];
            for e in 0..d {
                let code = (row[e / 2] >> (4 * (e % 2))) & 0xf;
                let s = half::f16::from_le_bytes([
                    row[d / 2 + e / 32 * 2],
                    row[d / 2 + e / 32 * 2 + 1],
                ])
                .to_f32();
                let x = want[t * d + e];
                let got = (f32::from(code) - 8.0) * s;
                assert!(
                    (got - x).abs() <= 0.5 * s + 1e-3 * x.abs() + 1e-6,
                    "token {t} elem {e}: {got} vs {x} (scale {s})"
                );
            }
        }

        // Prefill (all 96 tokens, WMMA) and decode (the last 8 tokens) attention per cache kind.
        let run = |kc: &Tensor, vc: &Tensor, queries: usize| -> Result<Tensor> {
            let q = q.narrow(0, tokens - queries, queries)?;
            let layout = k::TokenLayout::from_host(&[(0, queries)], &[tokens], queries, &rocm)?;
            let (selected, n_selected) = k::qsa_select_all(&layout, &paged, &shape, &rocm)?;
            k::kv_write_hnd(&kx, &vx, kc, vc, &slots, None, None)?;
            let (out, _) = k::qsa_attention(k::QsaAttentionArgs {
                q: &q,
                key_cache: kc,
                value_cache: vc,
                layout: &layout,
                paged: &paged,
                selected: &selected,
                n_selected: &n_selected,
                shape: &shape,
                n_kv_heads: kv_heads,
                scale: 1.0 / 16.0,
                gate: None,
                kv_scales: (1.0, 1.0),
            })?;
            out.to_dtype(DType::F32)
        };
        let rel = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let num = (a - b)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let den = b.sqr()?.sum_all()?.to_scalar::<f32>()?;
            Ok((num / den).sqrt())
        };
        for queries in [tokens, 8] {
            let reference = run(&cache(DType::BF16, d)?, &cache(DType::BF16, d)?, queries)?;
            let fp8 = rel(
                &run(
                    &cache(DType::F8E4M3, d)?,
                    &cache(DType::F8E4M3, d)?,
                    queries,
                )?,
                &reference,
            )?;
            let fp8_i4v = rel(
                &run(
                    &cache(DType::F8E4M3, d)?,
                    &cache(DType::U8, i4_row)?,
                    queries,
                )?,
                &reference,
            )?;
            let i4 = rel(
                &run(
                    &cache(DType::U8, i4_row)?,
                    &cache(DType::U8, i4_row)?,
                    queries,
                )?,
                &reference,
            )?;
            eprintln!("{queries} queries: rel rms vs bf16 cache: fp8 {fp8:.4}, fp8 K + 4-bit V {fp8_i4v:.4}, 4-bit {i4:.4}");
            assert!(
                fp8 < 0.1 && fp8_i4v < 0.2 && i4 < 0.25,
                "{queries} queries: {fp8} {fp8_i4v} {i4}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_partial_rope_matches_cpu() -> Result<()> {
        let rocm = Device::new_rocm(0)?;
        let x = Tensor::randn(0f32, 1.0, (2, 3, 4, 128), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let cos = Tensor::randn(0f32, 1.0, (2, 3, 32), &Device::Cpu)?;
        let sin = Tensor::randn(0f32, 1.0, (2, 3, 32), &Device::Cpu)?;
        let expected = rope_partial_neox(&x, &cos, &sin)?.to_dtype(DType::F32)?;
        let got = rope_partial_neox(
            &x.to_device(&rocm)?,
            &cos.to_device(&rocm)?,
            &sin.to_device(&rocm)?,
        )?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?;
        let diff = (expected - got)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-2, "max diff {diff}");
        Ok(())
    }

    #[test]
    fn mode_is_dense_until_the_budget_is_exceeded() {
        let qsa = QsaConfig {
            n_heads: 4,
            head_dim: 128,
            budget: 2048,
            compress_ratio: 4,
        };
        assert_eq!(QsaMode::for_max_kv_len(&qsa, 2051), QsaMode::Dense);
        assert_eq!(
            QsaMode::for_max_kv_len(&qsa, 2052),
            QsaMode::Sparse { max_blocks: 513 }
        );
    }
}
