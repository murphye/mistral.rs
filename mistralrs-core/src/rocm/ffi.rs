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
}
