use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use super::config::TextConfig;

/// Low-rank gated mixer over the `hc` residual streams; it stands in for every layernorm.
pub(super) struct GatedResidual {
    // (1 + w) in f32, [hc * hidden]
    pub(super) norm_weight: Tensor,
    pub(super) down: Arc<dyn QuantMethod>,
    pub(super) up: Arc<dyn QuantMethod>,
    pub(super) inject: Option<Arc<dyn QuantMethod>>,
    hc: usize,
    hidden: usize,
    eps: f64,
    /// The shared q8 activation seed the last fused mix left (ROCm decode), for the consumer
    /// that runs other launches first to renew.
    last_seed: std::sync::atomic::AtomicU64,
}

impl GatedResidual {
    pub(super) fn load(cfg: &TextConfig, vb: ShardedVarBuilder, with_inject: bool) -> Result<Self> {
        let hc_hidden = cfg.hc_hidden_size();
        let norm_weight = (vb
            .pp("hc_norm")
            .get(hc_hidden, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;
        let down = ReplicatedLayer::new(
            hc_hidden,
            cfg.hc_lowrank,
            &cfg.quantization_config,
            false,
            vb.pp("input_mix_weight_down"),
        )?;
        let up = ReplicatedLayer::new(
            cfg.hc_lowrank,
            hc_hidden,
            &cfg.quantization_config,
            false,
            vb.pp("input_mix_weight_up"),
        )?;
        let inject = with_inject
            .then(|| {
                ReplicatedLayer::new(
                    hc_hidden,
                    cfg.hc_count,
                    &cfg.quantization_config,
                    false,
                    vb.pp("block_inject_weight"),
                )
            })
            .transpose()?;
        Ok(Self {
            norm_weight,
            down,
            up,
            inject,
            hc: cfg.hc_count,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
            last_seed: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub(super) fn norm(&self, res: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return crate::cuda::qwen4_exp::hc_norm(res, &self.norm_weight, self.hc, self.eps);
        }
        #[cfg(feature = "rocm")]
        if res.device().is_rocm() {
            return crate::rocm::qwen4_exp::hc_norm(res, &self.norm_weight, self.hc, self.eps);
        }
        grouped_rms_norm(res, &self.norm_weight, self.hc, self.eps)
    }

    /// The block input: mean over streams of `xn * sigmoid(up(silu(down(xn) / hc)))`.
    pub(super) fn mix(&self, xn: &Tensor) -> Result<Tensor> {
        let lo = self.down.forward(xn)?;
        #[cfg(feature = "rocm")]
        let lo = if lo.device().is_rocm() {
            crate::rocm::moe_glue::act_scaled(&lo, 1.0 / self.hc as f32, 0)?
        } else {
            candle_nn::ops::silu(&(lo / self.hc as f64)?)?
        };
        #[cfg(not(feature = "rocm"))]
        let lo = candle_nn::ops::silu(&(lo / self.hc as f64)?)?;
        let gate = self.up.forward(&lo)?;
        #[cfg(feature = "cuda")]
        if xn.device().is_cuda() {
            return crate::cuda::qwen4_exp::hc_mix(xn, &gate, self.hc);
        }
        #[cfg(feature = "rocm")]
        if xn.device().is_rocm() {
            return crate::rocm::qwen4_exp::hc_mix(xn, &gate, self.hc);
        }
        let mut shape = xn.dims().to_vec();
        let last = shape.len() - 1;
        shape[last] = self.hc;
        shape.push(self.hidden);
        let gated = (xn * candle_nn::ops::sigmoid(&gate)?)?.reshape(shape)?;
        gated
            .to_dtype(DType::F32)?
            .mean(D::Minus2)?
            .to_dtype(xn.dtype())
    }

    /// [`Self::mix`] and [`Self::inject`] together. On ROCm decode batches both run as two
    /// fused launches over the quantized weights instead of eight small ones.
    pub(super) fn mix_inject(&self, xn: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "rocm")]
        if let Some(fused) = self.mix_inject_fused(xn)? {
            return Ok(fused);
        }
        Ok((self.mix(xn)?, self.inject(xn)?))
    }

    #[cfg(feature = "rocm")]
    fn mix_inject_fused(&self, xn: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
        use candle_core::quantized::rocm::{qgemv_gate_mean_seeded, qgemv_pair};
        let Some(inject) = &self.inject else {
            return Ok(None);
        };
        let (Some(down), Some(up), Some(inject)) =
            (self.down.get_qtensor(), self.up.get_qtensor(), inject.get_qtensor())
        else {
            return Ok(None);
        };
        if !xn.device().is_rocm() {
            return Ok(None);
        }
        let Some((lo, logits)) = qgemv_pair(xn, &down, Some(1.0 / self.hc as f32), &inject)? else {
            return Ok(None);
        };
        // The mixed activations feed quantized matmuls next (the GDN or QSA projections, the
        // MoE experts); the kernel leaves their q8_1 form for those to pick up.
        let Some((h, seed)) = qgemv_gate_mean_seeded(&lo, &up, xn, self.hc)? else {
            return Ok(None);
        };
        self.last_seed
            .store(seed.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
        Ok(Some((h, logits)))
    }

    /// The shared q8 activation seed left by the last [`Self::mix_inject`], once.
    pub(super) fn take_seed(&self) -> Option<u64> {
        match self.last_seed.swap(0, std::sync::atomic::Ordering::Relaxed) {
            0 => None,
            seed => Some(seed),
        }
    }

    /// Raw per-stream injection logits `[.., hc]`.
    pub(super) fn inject(&self, xn: &Tensor) -> Result<Tensor> {
        self.inject
            .as_ref()
            .expect("final hyper-connection mixer has no injection")
            .forward(xn)
    }

    /// `res + out * 2 * sigmoid(inject / hc)` per stream, plus the grouped norm `next` needs.
    pub(super) fn combine(
        &self,
        res: &Tensor,
        out: &Tensor,
        inject: &Tensor,
        next: Option<&GatedResidual>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let next = next.filter(|next| next.norm_weight.device().same_device(res.device()));
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return crate::cuda::qwen4_exp::hc_combine(
                res,
                out,
                inject,
                next.map(|next| &next.norm_weight),
                self.hc,
                self.eps,
            );
        }
        #[cfg(feature = "rocm")]
        if res.device().is_rocm() {
            return crate::rocm::qwen4_exp::hc_combine(
                res,
                out,
                inject,
                next.map(|next| &next.norm_weight),
                self.hc,
                self.eps,
            );
        }
        let dims = res.dims().to_vec();
        let mut stream_shape = dims.clone();
        let last = stream_shape.len() - 1;
        stream_shape[last] = self.hc;
        stream_shape.push(self.hidden);
        let weights =
            (candle_nn::ops::sigmoid(&(inject.to_dtype(DType::F32)? / self.hc as f64)?)? * 2.0)?;
        let injection = out
            .to_dtype(DType::F32)?
            .unsqueeze(D::Minus2)?
            .broadcast_mul(&weights.unsqueeze(D::Minus1)?)?;
        let res = (res.to_dtype(DType::F32)?.reshape(stream_shape)? + injection)?
            .reshape(dims)?
            .to_dtype(res.dtype())?;
        let xn = next.map(|next| next.norm(&res)).transpose()?;
        Ok((res, xn))
    }

    /// [`Self::combine`] where `out` came from a
    /// [`mistralrs_quant::distributed::DeferredReductions`] scope that skipped
    /// `skipped` all-reduces: one skipped reduction makes `out` this rank's partial sum, which
    /// the ROCm kernel reduces over peer memory as it combines, and which is otherwise reduced
    /// through `reducer` first.
    pub(super) fn combine_reduced(
        &self,
        res: &Tensor,
        out: &Tensor,
        inject: &Tensor,
        next: Option<&GatedResidual>,
        reducer: &mistralrs_quant::SumAllReduce,
        skipped: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        match skipped {
            0 => return self.combine(res, out, inject, next),
            1 => {}
            n => candle_core::bail!("{n} reductions deferred where one partial sum was expected"),
        }
        #[cfg(feature = "rccl")]
        if res.device().is_rocm() {
            let next = next.filter(|next| next.norm_weight.device().same_device(res.device()));
            if let Some(peer) = reducer.peer_raw_handles() {
                if crate::rocm::qwen4_exp::hc_combine_ar_supports(out, self.hc, &peer) {
                    return crate::rocm::qwen4_exp::hc_combine_ar(
                        res,
                        out,
                        inject,
                        next.map(|next| &next.norm_weight),
                        self.hc,
                        self.eps,
                        &peer,
                    );
                }
            }
        }
        let out = reducer.sum_all_reduce(&out.contiguous()?)?;
        self.combine(res, &out, inject, next)
    }
}

/// RMSNorm of each `hidden`-wide group of the last dim, times a per-group weight.
pub(super) fn grouped_rms_norm(
    x: &Tensor,
    weight: &Tensor,
    groups: usize,
    eps: f64,
) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let width = dims[dims.len() - 1];
    let hidden = width / groups;
    let mut grouped = dims.clone();
    let last = grouped.len() - 1;
    grouped[last] = groups;
    grouped.push(hidden);
    let xf = x.to_dtype(DType::F32)?.reshape(grouped)?;
    let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + eps)?
        .sqrt()?
        .recip()?;
    let normed = xf.broadcast_mul(&inv)?.reshape(dims)?;
    normed
        .broadcast_mul(&weight.to_device(x.device())?)?
        .to_dtype(x.dtype())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::grouped_rms_norm;

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    #[test]
    fn fused_hyper_connection_kernels_match_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (tokens, hc, hidden) = (5, 4, 96);
        let res =
            Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(DType::BF16)?;
        let weight = Tensor::randn(1f32, 0.1, hc * hidden, &device)?;
        let eps = 1e-6;

        let fused = crate::cuda::qwen4_exp::hc_norm(&res, &weight, hc, eps)?;
        let reference = grouped_rms_norm(&res, &weight, hc, eps)?;
        assert!(max_diff(&fused, &reference)? < 0.05);

        let gate =
            Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(DType::BF16)?;
        let mixed = crate::cuda::qwen4_exp::hc_mix(&fused, &gate, hc)?;
        let reference = (fused.to_dtype(DType::F32)?
            * candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
        .reshape((tokens, hc, hidden))?
        .mean(1)?;
        assert!(max_diff(&mixed, &reference)? < 0.02);

        let out = Tensor::randn(0f32, 1.0, (tokens, hidden), &device)?.to_dtype(DType::BF16)?;
        let inject = Tensor::randn(0f32, 1.0, (tokens, hc), &device)?.to_dtype(DType::BF16)?;
        let (combined, next) =
            crate::cuda::qwen4_exp::hc_combine(&res, &out, &inject, Some(&weight), hc, eps)?;
        let w = (candle_nn::ops::sigmoid(&(inject.to_dtype(DType::F32)? / hc as f64)?)? * 2.0)?;
        let reference = (res.to_dtype(DType::F32)?.reshape((tokens, hc, hidden))?
            + out
                .to_dtype(DType::F32)?
                .unsqueeze(1)?
                .broadcast_mul(&w.unsqueeze(2)?)?)?
        .reshape((tokens, hc * hidden))?;
        assert!(max_diff(&combined, &reference)? < 0.05);
        let next_reference = grouped_rms_norm(&combined, &weight, hc, eps)?;
        assert!(max_diff(&next.unwrap(), &next_reference)? < 0.05);
        Ok(())
    }
}
