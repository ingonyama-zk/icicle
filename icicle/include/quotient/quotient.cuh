#pragma once
#ifndef QUOTIENT_H
#define QUOTIENT_H

#include <cstdint>
#include "gpu-utils/device_context.cuh"

namespace quotient {
    template <typename QP, typename QF>
    struct ColumnSampleBatch {
        QP *point;
        uint32_t *columns;
        QF *values;
        uint32_t size;
    };

    struct QuotientConfig {
        device_context::DeviceContext ctx;
        bool are_columns_on_device;
        bool are_sample_points_on_device;
        bool are_results_on_device;
        bool is_async;
    };

    static QuotientConfig
    default_quotient_config(const device_context::DeviceContext& ctx = device_context::get_default_device_context())
    {
        QuotientConfig config = {
        ctx,
        false,
        false,
        false,
        false
        };
        return config;
    }

    template <typename QP, typename QF, typename CF, typename F, typename P>
    cudaError_t accumulate_quotients(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_log_size,
        F *columns, // 2d number_of_columns * domain_size elements
        uint32_t number_of_columns,
        QF &random_coefficient,
        ColumnSampleBatch<QP, QF> *samples,
        uint32_t sample_size,
        uint32_t flattened_line_coeffs_size,
        QuotientConfig &cfg,
        QF *result
    );
}
#endif