#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"

using namespace field_config; // Defines Zq and PolyRing for the active configuration

namespace icicle {

  //==============================================================================
  // vector_mul<PolyRing>
  //
  // Performs element-wise multiplication of a vector of polynomials (PolyRing)
  // with a vector of Zq scalars:
  //
  //   result[i] = poly_vec[i] * scalar_vec[i]  for i in [0, size)
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
    scalar_cfg.batch_size = size; // number of polynomials
    scalar_cfg.is_a_on_device = config.is_b_on_device;
    scalar_cfg.is_b_on_device = config.is_a_on_device;

    return scalar_mul_vec(input_scalars, input_vectors, PolyRing::d, scalar_cfg, output_vectors);
  }

  //==============================================================================
  // vector_sum<PolyRing>
  //
  // Reduces a vector of PolyRing elements into a single PolyRing by summing
  // coefficients across the vector:
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
    sum_cfg.batch_size = PolyRing::d;
    sum_cfg.columns_batch = true; // Sum across vector dimension for each coefficient

    return vector_sum(zq_in, size, sum_cfg, zq_out);
  }

} // namespace icicle