#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"

using namespace field_config; // Defines Zq, Rq, Tq for the active configuration

namespace icicle {

  //==============================================================================
  // Helper: polyring_vector_mul<P>
  // Performs element-wise multiplication: P[i] * Zq[i] -> P[i]
  // where P is a polynomial ring (e.g., Rq or Tq)
  //==============================================================================
  template <typename P>
  static eIcicleError
  polyring_vector_mul(const P* poly_vec, const Zq* scalar_vec, uint64_t size, const VecOpsConfig& config, P* result_vec)
  {
    if (config.columns_batch || config.batch_size != 1) {
      ICICLE_LOG_ERROR << "vector_mul<" << typeid(P).name()
                       << ", Zq> only supports batch_size == 1 and non-column layout";
      return eIcicleError::INVALID_ARGUMENT;
    }

    const Zq* input_vectors = reinterpret_cast<const Zq*>(poly_vec);
    const Zq* input_scalars = scalar_vec;
    Zq* output_vectors = reinterpret_cast<Zq*>(result_vec);

    VecOpsConfig scalar_cfg = config;
    scalar_cfg.batch_size = size;
    scalar_cfg.is_a_on_device = config.is_b_on_device;
    scalar_cfg.is_b_on_device = config.is_a_on_device;

    return scalar_mul_vec(input_scalars, input_vectors, P::d, scalar_cfg, output_vectors);
  }

  //==============================================================================
  // Specialization: vector_mul<Rq, Zq, Rq>
  //==============================================================================
  template <>
  eIcicleError vector_mul(const Rq* rq_vec, const Zq* zq_vec, uint64_t size, const VecOpsConfig& config, Rq* rq_out)
  {
    return polyring_vector_mul<Rq>(rq_vec, zq_vec, size, config, rq_out);
  }

  //==============================================================================
  // Specialization: vector_mul<Tq, Zq, Tq>
  //==============================================================================
  template <>
  eIcicleError vector_mul(const Tq* tq_vec, const Zq* zq_vec, uint64_t size, const VecOpsConfig& config, Tq* tq_out)
  {
    return polyring_vector_mul<Tq>(tq_vec, zq_vec, size, config, tq_out);
  }

  //==============================================================================
  // Helper: polyring_vector_sum<P>
  // Reduces a vector of polynomials P to a single P by summing coefficients across the vector.
  // Supports only batch_size == 1 and row-major layout.
  //==============================================================================
  template <typename P>
  static eIcicleError polyring_vector_sum(const P* vec, uint64_t size, const VecOpsConfig& config, P* output)
  {
    if (config.columns_batch || config.batch_size != 1) {
      ICICLE_LOG_ERROR << "vector_sum<" << typeid(P).name() << "> requires batch_size == 1 and row-major layout";
      return eIcicleError::INVALID_ARGUMENT;
    }

    const Zq* zq_vec = reinterpret_cast<const Zq*>(vec);
    Zq* zq_out = reinterpret_cast<Zq*>(output);

    VecOpsConfig sum_cfg = config;
    sum_cfg.batch_size = P::d;
    sum_cfg.columns_batch = true;

    return vector_sum(zq_vec, size, sum_cfg, zq_out);
  }

  //==============================================================================
  // Specialization: vector_sum<Rq>
  //==============================================================================
  template <>
  eIcicleError vector_sum(const Rq* vec, uint64_t size, const VecOpsConfig& config, Rq* output)
  {
    return polyring_vector_sum<Rq>(vec, size, config, output);
  }

  //==============================================================================
  // Specialization: vector_sum<Tq>
  //==============================================================================
  template <>
  eIcicleError vector_sum(const Tq* vec, uint64_t size, const VecOpsConfig& config, Tq* output)
  {
    return polyring_vector_sum<Tq>(vec, size, config, output);
  }

} // namespace icicle