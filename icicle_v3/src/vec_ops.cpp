#include "icicle/vec_ops.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** ADD ***********************************/
  ICICLE_DISPATCHER_INST(VectorAddDispatcher, vector_add, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_add)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t n, const VecOpsConfig& config, scalar_t* output)
  {
    return VectorAddDispatcher::execute(vec_a, vec_b, n, config, output);
  }

  template <>
  eIcicleError
  vector_add(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t n, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_add)(vec_a, vec_b, n, config, output);
  }

  /*********************************** SUB ***********************************/
  ICICLE_DISPATCHER_INST(VectorSubDispatcher, vector_sub, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_sub)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t n, const VecOpsConfig& config, scalar_t* output)
  {
    return VectorSubDispatcher::execute(vec_a, vec_b, n, config, output);
  }

  template <>
  eIcicleError
  vector_sub(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t n, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_sub)(vec_a, vec_b, n, config, output);
  }

  /*********************************** MUL ***********************************/
  ICICLE_DISPATCHER_INST(VectorMulDispatcher, vector_mul, scalarVectorOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_mul)(
    const scalar_t* vec_a, const scalar_t* vec_b, uint64_t n, const VecOpsConfig& config, scalar_t* output)
  {
    return VectorMulDispatcher::execute(vec_a, vec_b, n, config, output);
  }

  template <>
  eIcicleError
  vector_mul(const scalar_t* vec_a, const scalar_t* vec_b, uint64_t n, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_mul)(vec_a, vec_b, n, config, output);
  }

  /*********************************** GENERATE SCALARS ***********************************/

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, generate_scalars)(scalar_t* host_scalars, uint64_t size)
  {
    scalar_t::rand_host_many(host_scalars, size);
    return eIcicleError::SUCCESS;
  }

  template <>
  eIcicleError generate_scalars(scalar_t* host_scalars, uint64_t size)
  {
    return CONCAT_EXPAND(FIELD, generate_scalars)(host_scalars, size);
  }

  /*********************************** CONVERT MONTGOMERY ***********************************/

  ICICLE_DISPATCHER_INST(ScalarConvertMontgomeryDispatcher, scalar_convert_montgomery, scalarConvertMontgomeryImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(
    scalar_t* scalars, uint64_t size, bool is_into, const VecOpsConfig& config)
  {
    return ScalarConvertMontgomeryDispatcher::execute(scalars, size, is_into, config);
  }

  template <>
  eIcicleError scalar_convert_montgomery(scalar_t* scalars, uint64_t size, bool is_into, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(scalars, size, is_into, config);
  }

} // namespace icicle