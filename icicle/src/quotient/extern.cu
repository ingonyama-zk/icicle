#include "fields/field_config.cuh"

using namespace field_config;

#include "quotient.cu"
#include "utils/utils.h"

namespace quotient {
    extern "C" cudaError_t CONCAT_EXPAND(FIELD, accumulate_quotients_cuda)(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_log_size,
        scalar_t *columns, // 2d number_of_columns * domain_size elements
        uint32_t number_of_columns,
        q_extension_t random_coefficient,
        ColumnSampleBatch<secure_point_t, q_extension_t> *samples,
        uint32_t sample_size,
        uint32_t flattened_line_coeffs_size,
        QuotientConfig &cfg,
        q_extension_t *result
    ) {
        return accumulate_quotients<secure_point_t, q_extension_t, c_extension_t, scalar_t, point_t>(
            half_coset_initial_index,
            half_coset_step_size,
            domain_log_size,
            columns,
            number_of_columns,
            random_coefficient,
            samples,
            sample_size,
            flattened_line_coeffs_size,
            cfg,
            result
        );
    }
}