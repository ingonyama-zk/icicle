
#include "icicle/vec_ops/vec_ops.h"
#include "icicle/errors.h"
#include "icicle/device.h"

using namespace icicle;

IcicleError CpuVectorAdd(const Device& device, const int* vec_a, const int* vec_b, int n, int* output)
{
  for (int i = 0; i < n; ++i) {
    output[i] = vec_a[i] + vec_b[i];
  }
  return IcicleError::SUCCESS;
}

REGISTER_VECTOR_ADD_BACKEND("CPU", CpuVectorAdd);