#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"

using namespace field_config; // Defines Zq and PolyRing for the active configuration

namespace icicle {

  //==============================================================================
  // vector_mul<PolyRing, Zq>
  //
  // Performs element-wise multiplication of PolyRing[i] * Zq[i] → PolyRing[i],
  // for i in [0, size). Each polynomial is multiplied by a scalar.
  //
  // Only supports row-major layout with batch_size == 1.
  //==============================================================================
  template <>
  eIcicleError vector_mul(
    const PolyRing* poly_vec, const Zq* scalar_vec, uint64_t size, const VecOpsConfig& config, PolyRing* result_vec)
  {
    if (config.columns_batch || config.batch_size != 1) {
      ICICLE_LOG_ERROR << "vector_mul<PolyRing, Zq> requires batch_size == 1 and row-major layout";
      return eIcicleError::INVALID_ARGUMENT;
    }

    const Zq* input_vectors = reinterpret_cast<const Zq*>(poly_vec);
    const Zq* input_scalars = scalar_vec;
    Zq* output_vectors = reinterpret_cast<Zq*>(result_vec);

    VecOpsConfig scalar_cfg = config;
    scalar_cfg.batch_size = size; // number of PolyRing elements
    scalar_cfg.is_a_on_device = config.is_b_on_device;
    scalar_cfg.is_b_on_device = config.is_a_on_device;

    return scalar_mul_vec(input_scalars, input_vectors, PolyRing::d, scalar_cfg, output_vectors);
  }

  //==============================================================================
  // vector_mul<PolyRing, PolyRing>
  //
  // Performs element-wise multiplication of PolyRing[i] * PolyRing[i] → PolyRing[i],
  // for i in [0, size). Assumes inputs are in the evaluation domain (NTT).
  //
  // Only supports row-major layout with batch_size == 1.
  //==============================================================================
  template <>
  eIcicleError
  vector_mul(const PolyRing* vec_a, const PolyRing* vec_b, uint64_t size, const VecOpsConfig& config, PolyRing* vec_res)
  {
    if (config.columns_batch || config.batch_size != 1) {
      ICICLE_LOG_ERROR << "vector_mul<PolyRing, PolyRing> requires batch_size == 1 and row-major layout";
      return eIcicleError::INVALID_ARGUMENT;
    }

    const Zq* a_zq = reinterpret_cast<const Zq*>(vec_a);
    const Zq* b_zq = reinterpret_cast<const Zq*>(vec_b);
    Zq* res_zq = reinterpret_cast<Zq*>(vec_res);

    VecOpsConfig inner_cfg = config;
    inner_cfg.batch_size = size; // number of PolyRing elements

    return vector_mul(a_zq, b_zq, PolyRing::d, inner_cfg, res_zq);
  }

  //==============================================================================
  // vector_sum<PolyRing>
  //
  // Reduces a vector of PolyRing elements into a single PolyRing by summing
  // corresponding coefficients across the vector:
  //
  //   output = sum_{i=0}^{size-1} input_vec[i]
  //
  // Only supports row-major layout with batch_size == 1.
  //==============================================================================
  template <>
  eIcicleError vector_sum(const PolyRing* input_vec, uint64_t size, const VecOpsConfig& config, PolyRing* output)
  {
    if (config.columns_batch || config.batch_size != 1) {
      ICICLE_LOG_ERROR << "vector_sum<PolyRing> requires batch_size == 1 and row-major layout";
      return eIcicleError::INVALID_ARGUMENT;
    }

    const Zq* zq_in = reinterpret_cast<const Zq*>(input_vec);
    Zq* zq_out = reinterpret_cast<Zq*>(output);

    VecOpsConfig sum_cfg = config;
    sum_cfg.batch_size = PolyRing::d; // sum over all elements in each coefficient slot
    sum_cfg.columns_batch = true;     // interpret each coefficient index as a column

    return vector_sum(zq_in, size, sum_cfg, zq_out);
  }

  template <>
  eIcicleError matrix_mult(
    const PolyRing* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const PolyRing* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    PolyRing* mat_out)
  {
    const Zq* a = reinterpret_cast<const Zq*>(mat_a);
    const Zq* b = reinterpret_cast<const Zq*>(mat_b);
    Zq* c = reinterpret_cast<Zq*>(mat_out);
    auto degree = PolyRing::d;

    return poly_ring_matrix_mult(degree, a, nof_rows_a, nof_cols_a, b, nof_rows_b, nof_cols_b, config, c);
  }

} // namespace icicle