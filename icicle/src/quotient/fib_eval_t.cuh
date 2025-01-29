#pragma once
#include "stwo.cuh"

class fib_eval_t {
    uint32_t constraint_idx;
    const uint32_t random_coeff_powers_size;
    const qm31_t *random_coeff_powers;

    uint32_t trace_idx;
    const uint32_t trace_size;
    const m31_t *trace;

    qm31_t row_res;

    __host__ __device__ m31_t get_next_eval() {
        return trace[trace_idx++];
    }

    __host__ __device__ void add_constraint(const m31_t &constraint) {
        row_res = row_res + random_coeff_powers[constraint_idx++] * constraint;
    }

public:
    __host__ __device__ explicit fib_eval_t(
        const qm31_t *random_coeff_powers,
        const m31_t *trace,
        const uint32_t random_coeff_powers_size,
        const uint32_t trace_size
    ): constraint_idx(0),
       random_coeff_powers_size(random_coeff_powers_size),
       random_coeff_powers(random_coeff_powers),
       trace_idx(0),
       trace_size(trace_size),
       trace(trace),
       row_res(qm31_t::zero()) {}

    __host__ __device__ void evaluate() {
        m31_t a = get_next_eval();
        m31_t b = get_next_eval();
        for (uint32_t i = 0; i < trace_size - 2; i++) {
            const m31_t c = get_next_eval();
            add_constraint(c - a + b);
            a = b;
            b = c;
        }
    }

    __host__ __device__ qm31_t get_row_res() {
        return row_res;
    }
};
