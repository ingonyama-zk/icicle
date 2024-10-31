#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** REDUCE PRODUCT ************************/
  ICICLE_DISPATCHER_INST(VectorProductDispatcher, vector_product, VectorReduceOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_product)(
    const scalar_t* vec_a, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorProductDispatcher::execute(vec_a, size, *config, output);
  }

  template <>
  eIcicleError vector_product(const scalar_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_product)(vec_a, size, &config, output);
  }

  /*********************************** REDUCE SUM ****************************/
  ICICLE_DISPATCHER_INST(VectorSumDispatcher, vector_sum, VectorReduceOpImpl);

  extern "C" eIcicleError
  CONCAT_EXPAND(FIELD, vector_sum)(const scalar_t* vec_a, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorSumDispatcher::execute(vec_a, size, *config, output);
  }

  template <>
  eIcicleError vector_sum(const scalar_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_sum)(vec_a, size, &config, output);
  }

  /*********************************** ADD ***********************************/
  ICICLE_DISPATCHER_INST(VectorAddDispatcher, vector_add, vectorVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_add)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorAddDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_add(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_add)(vec_a, vec_b, size, &config, output);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(VectorAddExtFieldDispatcher, extension_vector_add, extFieldVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_vector_add)(
    const extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig* config, extension_t* output)
  {
    return VectorAddExtFieldDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_add(
    const extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig& config, extension_t* output)
  {
    return CONCAT_EXPAND(FIELD, extension_vector_add)(vec_a, vec_b, size, &config, output);
  }
#endif // EXT_FIELD

  /*********************************** ACCUMULATE ***********************************/
  ICICLE_DISPATCHER_INST(VectorAccumulateDispatcher, vector_accumulate, vectorVectorOpImplInplaceA);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_accumulate)(
    scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config)
  {
    return VectorAccumulateDispatcher::execute(vec_a, vec_b, size, *config);
  }

  template <>
  eIcicleError vector_accumulate(scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(FIELD, vector_accumulate)(vec_a, vec_b, size, &config);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(VectorAccumulateExtFieldDispatcher, extension_vector_accumulate, extFieldVectorOpImplInplaceA);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_vector_accumulate)(
    extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig* config)
  {
    return VectorAccumulateExtFieldDispatcher::execute(vec_a, vec_b, size, *config);
  }

  template <>
  eIcicleError
  vector_accumulate(extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(FIELD, extension_vector_accumulate)(vec_a, vec_b, size, &config);
  }
#endif // EXT_FIELD

  /*********************************** SUB ***********************************/
  ICICLE_DISPATCHER_INST(VectorSubDispatcher, vector_sub, vectorVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_sub)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorSubDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_sub(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_sub)(vec_a, vec_b, size, &config, output);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(VectorSubExtFieldDispatcher, extension_vector_sub, extFieldVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_vector_sub)(
    const extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig* config, extension_t* output)
  {
    return VectorSubExtFieldDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_sub(
    const extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig& config, extension_t* output)
  {
    return CONCAT_EXPAND(FIELD, extension_vector_sub)(vec_a, vec_b, size, &config, output);
  }
#endif // EXT_FIELD

  /*********************************** MUL ***********************************/
  ICICLE_DISPATCHER_INST(VectorMulDispatcher, vector_mul, vectorVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_mul)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorMulDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_mul(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_mul)(vec_a, vec_b, size, &config, output);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(VectorMulExtFieldDispatcher, extension_vector_mul, extFieldVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_vector_mul)(
    const extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig* config, extension_t* output)
  {
    return VectorMulExtFieldDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_mul(
    const extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig& config, extension_t* output)
  {
    return CONCAT_EXPAND(FIELD, extension_vector_mul)(vec_a, vec_b, size, &config, output);
  }
#endif // EXT_FIELD

  /*********************************** DIV ***********************************/
  ICICLE_DISPATCHER_INST(VectorDivDispatcher, vector_div, vectorVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_div)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return VectorDivDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError
  vector_div(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_div)(vec_a, vec_b, size, &config, output);
  }

  /*********************************** (Scalar + Vector) ELEMENT WISE ***********************************/
  ICICLE_DISPATCHER_INST(ScalarAddDispatcher, scalar_add_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_add_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarAddDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_add_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, scalar_add_vec)(scalar_a, vec_b, size, &config, output);
  }

  /*********************************** (Scalar - Vector) ELEMENT WISE ***********************************/
  ICICLE_DISPATCHER_INST(ScalarSubDispatcher, scalar_sub_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_sub_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarSubDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_sub_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, scalar_sub_vec)(scalar_a, vec_b, size, &config, output);
  }
  /*********************************** MUL BY SCALAR ***********************************/
  ICICLE_DISPATCHER_INST(ScalarMulDispatcher, scalar_mul_vec, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_mul_vec)(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarMulDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_mul_vec(
    const scalar_t* scalar_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, scalar_mul_vec)(scalar_a, vec_b, size, &config, output);
  }

  /*********************************** CONVERT MONTGOMERY ***********************************/

  ICICLE_DISPATCHER_INST(ScalarConvertMontgomeryDispatcher, scalar_convert_montgomery, scalarConvertMontgomeryImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(
    const scalar_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarConvertMontgomeryDispatcher::execute(input, size, is_to_montgomery, *config, output);
  }

  template <>
  eIcicleError convert_montgomery(
    const scalar_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(input, size, is_to_montgomery, &config, output);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(
    ExtFieldConvertMontgomeryDispatcher, extension_scalar_convert_montgomery, extFieldConvertMontgomeryImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_scalar_convert_montgomery)(
    const extension_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig* config, extension_t* output)
  {
    return ExtFieldConvertMontgomeryDispatcher::execute(input, size, is_to_montgomery, *config, output);
  }

  template <>
  eIcicleError convert_montgomery(
    const extension_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, extension_t* output)
  {
    return CONCAT_EXPAND(FIELD, extension_scalar_convert_montgomery)(input, size, is_to_montgomery, &config, output);
  }
#endif // EXT_FIELD

  /*********************************** BIT REVERSE ***********************************/

  ICICLE_DISPATCHER_INST(ScalarBitReverseDispatcher, scalar_bit_reverse, scalarBitReverseOpImpl)

  extern "C" eIcicleError
  CONCAT_EXPAND(FIELD, bit_reverse)(const scalar_t* input, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ScalarBitReverseDispatcher::execute(input, size, *config, output);
  }

  template <>
  eIcicleError bit_reverse(const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, bit_reverse)(input, size, &config, output);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(ExtFieldBitReverseDispatcher, extension_bit_reverse, extFieldBitReverseOpImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_bit_reverse)(
    const extension_t* input, uint64_t size, const VecOpsConfig* config, extension_t* output)
  {
    return ExtFieldBitReverseDispatcher::execute(input, size, *config, output);
  }

  template <>
  eIcicleError bit_reverse(const extension_t* input, uint64_t size, const VecOpsConfig& config, extension_t* output)
  {
    return CONCAT_EXPAND(FIELD, extension_bit_reverse)(input, size, &config, output);
  }
#endif // EXT_FIELD

  /*********************************** SLICE ***********************************/

  ICICLE_DISPATCHER_INST(ScalarSliceDispatcher, slice, scalarSliceOpImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, slice)(
    const scalar_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig* config,
    scalar_t* output)
  {
    return ScalarSliceDispatcher::execute(input, offset, stride, size_in, size_out, *config, output);
  }

  template <>
  eIcicleError slice(
    const scalar_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig& config,
    scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, slice)(input, offset, stride, size_in, size_out, &config, output);
  }

  // Deprecated API
  template <>
  eIcicleError slice(
    const scalar_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_out,
    const VecOpsConfig& config,
    scalar_t* output)
  {
    const auto size_in = offset + stride * (size_out - 1) + 1; // input should be at least that large
    ICICLE_LOG_WARNING << "slice api is deprecated and replace with new api. Use new slice api instead";
    if (config.batch_size != 1) {
      ICICLE_LOG_ERROR << "deprecated slice API does not support batch";
      return eIcicleError::INVALID_ARGUMENT;
    }
    return slice(input, offset, stride, size_in, size_out, config, output);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(ExtFieldSliceDispatcher, extension_slice, extFieldSliceOpImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_slice)(
    const extension_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size,
    const VecOpsConfig* config,
    extension_t* output)
  {
    return ExtFieldSliceDispatcher::execute(input, offset, stride, size, *config, output);
  }

  template <>
  eIcicleError slice(
    const extension_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size,
    const VecOpsConfig& config,
    extension_t* output)
  {
    return CONCAT_EXPAND(FIELD, extension_slice)(input, offset, stride, size, &config, output);
  }
#endif // EXT_FIELD

  /*********************************** HIGHEST sizeON ZERO IDX ***********************************/

  ICICLE_DISPATCHER_INST(ScalarHighestNonZeroIdxDispatcher, highest_non_zero_idx, scalarHighNonZeroIdxOpImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, highest_non_zero_idx)(
    const scalar_t* input, uint64_t size, const VecOpsConfig* config, int64_t* out_idx /*OUT*/)
  {
    return ScalarHighestNonZeroIdxDispatcher::execute(input, size, *config, out_idx);
  }

  template <>
  eIcicleError
  highest_non_zero_idx(const scalar_t* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx /*OUT*/)
  {
    return CONCAT_EXPAND(FIELD, highest_non_zero_idx)(input, size, &config, out_idx);
  }

  /*********************************** POLY EVAL ***********************************/

  ICICLE_DISPATCHER_INST(ScalarPolyEvalDispatcher, poly_eval, scalarPolyEvalImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, poly_eval)(
    const scalar_t* coeffs,
    uint64_t coeffs_size,
    const scalar_t* domain,
    uint64_t domain_size,
    const VecOpsConfig* config,
    scalar_t* evals /*OUT*/)
  {
    return ScalarPolyEvalDispatcher::execute(coeffs, coeffs_size, domain, domain_size, *config, evals);
  }

  template <>
  eIcicleError polynomial_eval(
    const scalar_t* coeffs,
    uint64_t coeffs_size,
    const scalar_t* domain,
    uint64_t domain_size,
    const VecOpsConfig& config,
    scalar_t* evals /*OUT*/)
  {
    return CONCAT_EXPAND(FIELD, poly_eval)(coeffs, coeffs_size, domain, domain_size, &config, evals);
  }

  /*********************************** POLY DIVISION ***********************************/

  ICICLE_DISPATCHER_INST(ScalarPolyDivDispatcher, poly_division, scalarPolyDivImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, poly_division)(
    const scalar_t* numerator,
    int64_t numerator_deg,
    uint64_t numerator_size,
    const scalar_t* denumerator,
    int64_t denumerator_deg,
    uint64_t denumerator_size,
    uint64_t q_size,
    uint64_t r_size,
    const VecOpsConfig& config,
    scalar_t* q_out /*OUT*/,
    scalar_t* r_out /*OUT*/)
  {
    return ScalarPolyDivDispatcher::execute(
      numerator, numerator_deg, numerator_size, denumerator, denumerator_deg, denumerator_size, q_size, r_size, config,
      q_out, r_out);
  }

  template <>
  eIcicleError polynomial_division(
    const scalar_t* numerator,
    int64_t numerator_deg,
    uint64_t numerator_size,
    const scalar_t* denumerator,
    int64_t denumerator_deg,
    uint64_t denumerator_size,
    uint64_t q_size,
    uint64_t r_size,
    const VecOpsConfig& config,
    scalar_t* q_out /*OUT*/,
    scalar_t* r_out /*OUT*/)
  {
    return CONCAT_EXPAND(FIELD, poly_division)(
      numerator, numerator_deg, numerator_size, denumerator, denumerator_deg, denumerator_size, q_size, r_size, config,
      q_out, r_out);
  }

  // Deprecated API
  template <>
  eIcicleError polynomial_division(
    const scalar_t* numerator,
    int64_t numerator_deg,
    const scalar_t* denumerator,
    int64_t denumerator_deg,
    const VecOpsConfig& config,
    scalar_t* q_out /*OUT*/,
    uint64_t q_size,
    scalar_t* r_out /*OUT*/,
    uint64_t r_size)
  {
    ICICLE_LOG_WARNING
      << "polynomial_division api is deprecated and replace with new api. Use new polynomial_division api instead";
    if (config.batch_size != 1) {
      ICICLE_LOG_ERROR << "deprecated polynomial_division API does not support batch";
      return eIcicleError::INVALID_ARGUMENT;
    }
    return polynomial_division(
      numerator, numerator_deg, numerator_deg + 1, denumerator, denumerator_deg, denumerator_deg + 1, q_size, r_size,
      config, q_out, r_out);
  }

} // namespace icicle