pub mod autotuner;
pub(crate) mod builtin_mtp;
pub mod cache;
pub mod config;
pub mod dflash;
pub mod driver;
pub(crate) mod gdn_transitions;
pub(crate) mod hybrid_state;
pub mod logging;
pub(crate) mod paged_rows;
pub mod policy;
pub mod proposer;
pub(crate) mod staging;
pub mod target;
pub mod verifier;

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use crate::cuda::speculative_rejection::CudaSparseRejectionWorkspace;
pub use config::{
    reserve_external_mtp_memory, reserve_external_mtp_memory_with_runtime, MtpConfig,
    MtpDraftSamplingMethod, MtpRuntimeConfig, SpeculativeConfig,
};
pub use dflash::DFlashDraftModel;
pub use logging::{SpeculativeAttachInfo, SpeculativeAttachKind};
pub use policy::{SpeculativeBatchObservation, SpeculativeBatchPlan, SpeculativeGraphPlan};
pub(crate) use proposer::SpeculativeTapRouting;
pub use proposer::{
    SparseSpeculativeProbs, SpeculativeCommitRow, SpeculativeKvCache, SpeculativePrefillCtx,
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposalDistribution,
    SpeculativeProposeBatchCtx, SpeculativeProposePreparation, SpeculativeProposePrepareCtx,
    SpeculativeProposer, SpeculativeTokens, TargetAttentionInputs, TargetTokenEmbedder,
};
pub use target::{
    SpeculativeGraphState, SpeculativePrefixCheckpointPolicy, SpeculativePrefixReplay,
    SpeculativeTargetMixin,
};

#[cfg(test)]
mod tests {
    use super::{MtpConfig, MtpDraftSamplingMethod};

    #[test]
    fn mtp_config_supports_public_struct_literals() {
        let config = MtpConfig {
            model: Some("assistant".to_string()),
            n_predict: Some(3),
            draft_sampling_method: MtpDraftSamplingMethod::Probabilistic,
            draft_lm_head_isq: None,
        };

        assert_eq!(config.n_predict, Some(3));
    }
}

/// Temporary phase timing: with `MISTRALRS_SPEC_TIMING` set, synchronize the GPUs and log
/// how long the phase that began at `start` took.
pub(crate) fn spec_timing(phase: &str, start: std::time::Instant) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("MISTRALRS_SPEC_TIMING").is_some()) {
        return;
    }
    let host = start.elapsed().as_secs_f64() * 1e3;
    #[cfg(feature = "rocm")]
    {
        static DEVICES: std::sync::OnceLock<Vec<candle_core::Device>> = std::sync::OnceLock::new();
        for device in DEVICES.get_or_init(|| {
            (0..2)
                .filter_map(|ordinal| candle_core::Device::new_rocm(ordinal).ok())
                .collect()
        }) {
            let _ = device.synchronize();
        }
    }
    eprintln!(
        "SPEC_TIMING {phase} {:.2}ms host {host:.2}ms",
        start.elapsed().as_secs_f64() * 1e3
    );
}
