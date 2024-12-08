#include "fields/field_config.cuh"

using namespace field_config;

#include "quotient.cu"
#include "utils/utils.h"

namespace quotient {
    extern "C" cudaError_t CONCAT_EXPAND(FIELD, accumulate_quotients)(
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
        domain_t domain = domain_t(domain_log_size);
        return accumulate_quotients<secure_point_t, q_extension_t, c_extension_t, scalar_t, point_t, domain_t>(
            domain,
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