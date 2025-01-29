#pragma once
#include <cstdint>
#include "/home/vhnat/repo/icicle/icicle/include/fields/stark_fields/m31.cuh"

using m31_t = m31::scalar_t;
using cm31_t = m31::c_extension_t;
using qm31_t = m31::q_extension_t;
using domain_t = m31::domain_t;
using point_t = m31::point_t;
using coset_t = m31::coset_t;

struct sub_trace_t {
    m31_t *trace = nullptr;
    uint32_t n_cols;
    uint32_t n_rows;

    sub_trace_t() = default;

    explicit sub_trace_t(const uint32_t n_cols, const uint32_t n_rows) : n_cols(n_cols), n_rows(n_rows) {
        cudaMallocManaged(&trace, n_cols * n_rows * sizeof(m31_t));
    }

    m31_t *get_row(const uint32_t row_idx) const {
        return trace + row_idx * n_cols;
    }
};

struct component_ctx_t {
    uint32_t domain_log_size;
    uint32_t eval_log_size;
    sub_trace_t preprocessing_trace;
    sub_trace_t execution_trace;
    sub_trace_t interaction_trace;

    void allocate_preprocessing_trace(const uint32_t n_cols, const uint32_t n_rows) {
        preprocessing_trace = sub_trace_t(n_cols, n_rows);
    }

    void allocate_execution_trace(const uint32_t n_cols, const uint32_t n_rows) {
        execution_trace = sub_trace_t(n_cols, n_rows);
    }

    void allocate_interaction_trace(const uint32_t n_cols, const uint32_t n_rows) {
        interaction_trace = sub_trace_t(n_cols, n_rows);
    }
};

struct prover_ctx_t {
    uint32_t total_constraints;
    qm31_t *random_coeff_powers = nullptr;
    qm31_t *composition_poly = nullptr;
};
