#include "icicle/vec_ops.h"
#include "icicle/dispatcher.h"

using namespace icicle;

/*********************************** ADD ***********************************/
ICICLE_DISPATCHER_INST(VectorAddDispatcher, vector_add, scalarVectorOpImpl);

extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_add)(
  const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
{
  return VectorAddDispatcher::execute(vec_a, vec_b, n, config, output);
}

/*********************************** SUB ***********************************/
ICICLE_DISPATCHER_INST(VectorSubDispatcher, vector_sub, scalarVectorOpImpl);

extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_sub)(
  const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
{
  return VectorSubDispatcher::execute(vec_a, vec_b, n, config, output);
}

/*********************************** MUL ***********************************/
ICICLE_DISPATCHER_INST(VectorMulDispatcher, vector_mul, scalarVectorOpImpl);

extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_mul)(
  const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
{
  return VectorMulDispatcher::execute(vec_a, vec_b, n, config, output);
}