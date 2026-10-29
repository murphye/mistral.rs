//! Launchers from the CUDA kernels in `src/cuda` compiled with hipcc for ROCm.
//! `stream` arguments take a `hipStream_t` cast to i64.
use core::ffi::c_void;

extern "C" {
    pub fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    pub fn warp_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    pub fn chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    pub fn causal_conv1d_update(
        x: *const c_void,
        weight: *const c_void,
        conv_state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );

    pub fn causal_conv1d_full(
        x: *const c_void,
        weight: *const c_void,
        conv_state_in: *const c_void,
        conv_state_out: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );

    pub fn gdn_rmsnorm_gated(
        x: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden_dim: i32,
        outer_dim_1: i32,
        outer_dim_2: i32,
        x_stride_0: i64,
        x_stride_1: i64,
        x_stride_2: i64,
        x_stride_3: i64,
        gate_stride_0: i64,
        gate_stride_1: i64,
        gate_stride_2: i64,
        gate_stride_3: i64,
        eps: f32,
        sigmoid_gate: i32,
        dtype: i32,
        stream: i64,
    );

    pub fn fused_gdn_gating(
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        total_elements: i32,
        num_heads: i32,
        dtype: i32,
        stream: i64,
    );

    pub fn gdn_prepare_recurrence(
        mixed_qkv: *const c_void,
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        g_out: *mut f32,
        beta_out: *mut f32,
        batch_size: i32,
        seq_len: i32,
        token_stride: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        tiled_v_heads: i32,
        dtype: i32,
        out_bf16: i32,
        stream: i64,
    );

    pub fn gdn_decode_recurrence_multi(
        mixed_qkv: *const c_void,
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        tiled_v_heads: i32,
        b_strides: *const i64,
        a_strides: *const i64,
        slot_indices: *const i32,
        dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    pub fn gdn_decode_recurrence(
        mixed_qkv: *const c_void,
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        tiled_v_heads: i32,
        b_batch_stride: i64,
        b_head_stride: i64,
        a_batch_stride: i64,
        a_head_stride: i64,
        slot_indices: *const i32,
        kernel_kind: i32,
        dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    pub fn moe_router_topk_f32(
        logits: *const c_void,
        weights: *mut c_void,
        ids: *mut c_void,
        selection_bias: *const c_void,
        expert_scale: *const c_void,
        nrows: i32,
        n_experts: i32,
        top_k: i32,
        score_mode: i32,
        weight_mode: i32,
        renormalize: bool,
        clamp_logits: bool,
        clamp_min: f32,
        clamp_max: f32,
        norm_min: f32,
        output_scale: f32,
        stream: i64,
    );
    pub fn moe_router_topk_bf16(
        logits: *const c_void,
        weights: *mut c_void,
        ids: *mut c_void,
        selection_bias: *const c_void,
        expert_scale: *const c_void,
        nrows: i32,
        n_experts: i32,
        top_k: i32,
        score_mode: i32,
        weight_mode: i32,
        renormalize: bool,
        clamp_logits: bool,
        clamp_min: f32,
        clamp_max: f32,
        norm_min: f32,
        output_scale: f32,
        stream: i64,
    );
    pub fn moe_router_topk_f16(
        logits: *const c_void,
        weights: *mut c_void,
        ids: *mut c_void,
        selection_bias: *const c_void,
        expert_scale: *const c_void,
        nrows: i32,
        n_experts: i32,
        top_k: i32,
        score_mode: i32,
        weight_mode: i32,
        renormalize: bool,
        clamp_logits: bool,
        clamp_min: f32,
        clamp_max: f32,
        norm_min: f32,
        output_scale: f32,
        stream: i64,
    );

    pub fn glu_act_mul_f32(gate: *const c_void, up: *const c_void, out: *mut c_void, n: i64, kind: i32, stream: i64);
    pub fn glu_act_mul_f16(gate: *const c_void, up: *const c_void, out: *mut c_void, n: i64, kind: i32, stream: i64);
    pub fn glu_act_mul_bf16(gate: *const c_void, up: *const c_void, out: *mut c_void, n: i64, kind: i32, stream: i64);
    pub fn moe_weighted_reduce_f32(ys: *const c_void, weights: *const f32, out: *mut c_void, tokens: i32, top_k: i32, hidden: i32, stream: i64);
    pub fn moe_weighted_reduce_f16(ys: *const c_void, weights: *const f32, out: *mut c_void, tokens: i32, top_k: i32, hidden: i32, stream: i64);
    pub fn moe_weighted_reduce_bf16(ys: *const c_void, weights: *const f32, out: *mut c_void, tokens: i32, top_k: i32, hidden: i32, stream: i64);
    pub fn rope_partial_neox_f32(x: *const c_void, cos: *const f32, sin: *const f32, out: *mut c_void, n: i64, heads: i32, d: i32, half: i32, stream: i64);
    pub fn rope_partial_neox_f16(x: *const c_void, cos: *const f32, sin: *const f32, out: *mut c_void, n: i64, heads: i32, d: i32, half: i32, stream: i64);
    pub fn rope_partial_neox_bf16(x: *const c_void, cos: *const f32, sin: *const f32, out: *mut c_void, n: i64, heads: i32, d: i32, half: i32, stream: i64);
    pub fn moe_append_shared_slot_f32(ids: *const u32, weights: *const f32, gate: *const c_void, ids_out: *mut u32, weights_out: *mut f32, tokens: i32, top_k: i32, shared_id: u32, stream: i64);
    pub fn moe_append_shared_slot_f16(ids: *const u32, weights: *const f32, gate: *const c_void, ids_out: *mut u32, weights_out: *mut f32, tokens: i32, top_k: i32, shared_id: u32, stream: i64);
    pub fn moe_append_shared_slot_bf16(ids: *const u32, weights: *const f32, gate: *const c_void, ids_out: *mut u32, weights_out: *mut f32, tokens: i32, top_k: i32, shared_id: u32, stream: i64);
    pub fn add_sigmoid_gated_f32(y: *const c_void, shared: *const c_void, gate: *const c_void, out: *mut c_void, n: i64, hidden: i32, stream: i64);
    pub fn add_sigmoid_gated_f16(y: *const c_void, shared: *const c_void, gate: *const c_void, out: *mut c_void, n: i64, hidden: i32, stream: i64);
    pub fn add_sigmoid_gated_bf16(y: *const c_void, shared: *const c_void, gate: *const c_void, out: *mut c_void, n: i64, hidden: i32, stream: i64);
    pub fn topk_large_packed_f32(input: *const c_void, block_values: *mut f32, block_indices: *mut u32, block_maxes: *mut f32, block_sums: *mut f32, packed_out: *mut f32, ncols: i32, k: i32, chunk_size: i32, nblocks: i32, inv_temperature: f32, stream: i64);
    pub fn topk_large_packed_f16(input: *const c_void, block_values: *mut f32, block_indices: *mut u32, block_maxes: *mut f32, block_sums: *mut f32, packed_out: *mut f32, ncols: i32, k: i32, chunk_size: i32, nblocks: i32, inv_temperature: f32, stream: i64);
    pub fn topk_large_packed_bf16(input: *const c_void, block_values: *mut f32, block_indices: *mut u32, block_maxes: *mut f32, block_sums: *mut f32, packed_out: *mut f32, ncols: i32, k: i32, chunk_size: i32, nblocks: i32, inv_temperature: f32, stream: i64);
    pub fn act_scaled_f32(x: *const c_void, out: *mut c_void, n: i64, scale: f32, kind: i32, stream: i64);
    pub fn act_scaled_f16(x: *const c_void, out: *mut c_void, n: i64, scale: f32, kind: i32, stream: i64);
    pub fn act_scaled_bf16(x: *const c_void, out: *mut c_void, n: i64, scale: f32, kind: i32, stream: i64);
    pub fn qsa_pack_keys_f32(k: *const c_void, raw_key: *const c_void, cos: *const c_void, sin: *const c_void, out: *mut c_void, batch: i32, heads: i32, seq: i32, d: i32, d_idx: i32, half: i32, raw_stride: i64, cs_batch_stride: i64, stream: i64);
    pub fn qsa_pack_keys_f16(k: *const c_void, raw_key: *const c_void, cos: *const c_void, sin: *const c_void, out: *mut c_void, batch: i32, heads: i32, seq: i32, d: i32, d_idx: i32, half: i32, raw_stride: i64, cs_batch_stride: i64, stream: i64);
    pub fn qsa_pack_keys_bf16(k: *const c_void, raw_key: *const c_void, cos: *const c_void, sin: *const c_void, out: *mut c_void, batch: i32, heads: i32, seq: i32, d: i32, d_idx: i32, half: i32, raw_stride: i64, cs_batch_stride: i64, stream: i64);
    pub fn topk_large_packed_batched_f32(input: *const c_void, inv_temperatures: *const f32, block_values: *mut f32, block_indices: *mut u32, block_maxes: *mut f32, block_sums: *mut f32, packed_out: *mut f32, nrows: i32, ncols: i32, k: i32, chunk_size: i32, nblocks: i32, stream: i64);
    pub fn topk_large_packed_batched_f16(input: *const c_void, inv_temperatures: *const f32, block_values: *mut f32, block_indices: *mut u32, block_maxes: *mut f32, block_sums: *mut f32, packed_out: *mut f32, nrows: i32, ncols: i32, k: i32, chunk_size: i32, nblocks: i32, stream: i64);
    pub fn topk_large_packed_batched_bf16(input: *const c_void, inv_temperatures: *const f32, block_values: *mut f32, block_indices: *mut u32, block_maxes: *mut f32, block_sums: *mut f32, packed_out: *mut f32, nrows: i32, ncols: i32, k: i32, chunk_size: i32, nblocks: i32, stream: i64);
    pub fn indexed_row_copy(src: *const c_void, dst: *mut c_void, rows: *const u32, n_rows: i64, row_elems: i64, elem_bytes: i32, stream: i64);
}
