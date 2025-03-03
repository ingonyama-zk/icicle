#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {
  /***********************************  RING RNS ************************/
  ICICLE_DISPATCHER_INST(VectorSumRingRnsDispatcher, ring_rns_vector_sum, ringRnsVectorReduceOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_sum)(
    const scalar_rns_t* vec_a, uint64_t size, const VecOpsConfig* config, scalar_rns_t* output)
  {
    return VectorSumRingRnsDispatcher::execute(vec_a, size, *config, output);
  }

  template <>
  eIcicleError vector_sum(const scalar_rns_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_sum)(vec_a, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(VectorAddRingRnsDispatcher, ring_rns_vector_add, ringRnsVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_add)(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return VectorAddRingRnsDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_add(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_add)(vec_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(VectorAccumulateRingRnsDispatcher, ring_rns_vector_accumulate, ringRnsVectorOpImplInplaceA);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_accumulate)(
    scalar_rns_t* vec_a, const scalar_rns_t* vec_b, uint64_t size, const VecOpsConfig* config)
  {
    return VectorAccumulateRingRnsDispatcher::execute(vec_a, vec_b, size, *config);
  }

  template <>
  eIcicleError
  vector_accumulate(scalar_rns_t* vec_a, const scalar_rns_t* vec_b, uint64_t size, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_accumulate)(vec_a, vec_b, size, &config);
  }

  ICICLE_DISPATCHER_INST(VectorSubRingRnsDispatcher, ring_rns_vector_sub, ringRnsVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_sub)(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return VectorSubRingRnsDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_sub(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_sub)(vec_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(VectorMulRingRnsDispatcher, ring_rns_vector_mul, ringRnsVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_mul)(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return VectorMulRingRnsDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_mul(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_mul)(vec_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(VectorMixedMulDispatcher, ring_rns_vector_mixed_mul, mixedVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_mixed_mul)(
    const scalar_rns_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, scalar_rns_t* output)
  {
    return VectorMixedMulDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_mul(
    const scalar_rns_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_mixed_mul)(vec_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(VectorDivRingRnsDispatcher, ring_rns_vector_div, ringRnsVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_div)(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return VectorDivRingRnsDispatcher::execute(vec_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError vector_div(
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_div)(vec_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(ScalarAddRingRnsDispatcher, ring_rns_scalar_add_vec, ringRnsVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_add_vec)(
    const scalar_rns_t* scalar_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return ScalarAddRingRnsDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_add_vec(
    const scalar_rns_t* scalar_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_add_vec)(scalar_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(ScalarSubRingRnsDispatcher, ring_rns_scalar_sub_vec, ringRnsVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_sub_vec)(
    const scalar_rns_t* scalar_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return ScalarSubRingRnsDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_sub_vec(
    const scalar_rns_t* scalar_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_sub_vec)(scalar_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(ScalarMulRingRnsDispatcher, ring_rns_scalar_mul_vec, ringRnsVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_mul_vec)(
    const scalar_rns_t* scalar_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return ScalarMulRingRnsDispatcher::execute(scalar_a, vec_b, size, *config, output);
  }

  template <>
  eIcicleError scalar_mul_vec(
    const scalar_rns_t* scalar_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_mul_vec)(scalar_a, vec_b, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(
    RingRnsConvertMontgomeryDispatcher, ring_rns_scalar_convert_montgomery, ringRnsConvertMontgomeryImpl)

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_convert_montgomery)(
    const scalar_rns_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig* config, scalar_rns_t* output)
  {
    return RingRnsConvertMontgomeryDispatcher::execute(input, size, is_to_montgomery, *config, output);
  }

  template <>
  eIcicleError convert_montgomery(
    const scalar_rns_t* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_scalar_convert_montgomery)(
      input, size, is_to_montgomery, &config, output);
  }

  ICICLE_DISPATCHER_INST(RingRnsBitReverseDispatcher, ring_rns_bit_reverse, ringRnsBitReverseOpImpl)

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_bit_reverse)(
    const scalar_rns_t* input, uint64_t size, const VecOpsConfig* config, scalar_rns_t* output)
  {
    return RingRnsBitReverseDispatcher::execute(input, size, *config, output);
  }

  template <>
  eIcicleError bit_reverse(const scalar_rns_t* input, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_bit_reverse)(input, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(RingRnsSliceDispatcher, ring_rns_slice, ringRnsSliceOpImpl)

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_slice)(
    const scalar_rns_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig* config,
    scalar_rns_t* output)
  {
    return RingRnsSliceDispatcher::execute(input, offset, stride, size_in, size_out, *config, output);
  }

  template <>
  eIcicleError slice(
    const scalar_rns_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig& config,
    scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_slice)(input, offset, stride, size_in, size_out, &config, output);
  }

  ICICLE_DISPATCHER_INST(VectorProductRingRnsDispatcher, ring_rns_vector_product, ringRnsVectorReduceOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_product)(
    const scalar_rns_t* vec_a, uint64_t size, const VecOpsConfig* config, scalar_rns_t* output)
  {
    return VectorProductRingRnsDispatcher::execute(vec_a, size, *config, output);
  }

  template <>
  eIcicleError
  vector_product(const scalar_rns_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_vector_product)(vec_a, size, &config, output);
  }

  /* RNS <--> direct conversion*/
  ICICLE_DISPATCHER_INST(ConvertToRnsDispatcher, convert_to_rns, ringConvertToRnsImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, convert_to_rns)(
    const scalar_t* input, uint64_t size, const VecOpsConfig* config, scalar_rns_t* output)
  {
    return ConvertToRnsDispatcher::execute(input, size, *config, output);
  }

  template <>
  eIcicleError convert_to_rns(const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, convert_to_rns)(input, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(ConvertFromRnsDispatcher, convert_from_rns, ringConvertFromRnsImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, convert_from_rns)(
    const scalar_rns_t* input, uint64_t size, const VecOpsConfig* config, scalar_t* output)
  {
    return ConvertFromRnsDispatcher::execute(input, size, *config, output);
  }

  template <>
  eIcicleError convert_from_rns(const scalar_rns_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, convert_from_rns)(input, size, &config, output);
  }

  ICICLE_DISPATCHER_INST(RnsExecuteProgramDispatcher, rns_execute_program, rnsProgramExecutionImpl)

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_execute_program)(
    scalar_rns_t** data_ptr,
    uint64_t nof_params,
    const Program<scalar_rns_t>* program,
    uint64_t size,
    const VecOpsConfig& config)
  {
    std::vector<scalar_rns_t*> data_vec;
    data_vec.reserve(nof_params);
    for (uint64_t i = 0; i < nof_params; i++) {
      if (data_ptr[i] == nullptr) { throw std::invalid_argument("Null pointer found in parameters"); }
      data_vec.push_back(data_ptr[i]);
    }
    return RnsExecuteProgramDispatcher::execute(data_vec, *program, size, config);
  }

  template <>
  eIcicleError execute_program(
    std::vector<scalar_rns_t*>& data, const Program<scalar_rns_t>& program, uint64_t size, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_execute_program)(data.data(), data.size(), &program, size, config);
  }

} // namespace icicle