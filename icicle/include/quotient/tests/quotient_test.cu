#include "fields/id.h"
#define FIELD_ID M31
#define DCCT

#include "fields/field_config.cuh"
typedef field_config::scalar_t test_scalar;
typedef field_config::c_extension_t test_ext;
typedef field_config::scalar_t test_data;

#include "fields/field.cuh"
#include "curves/projective.cuh"
#include <chrono>
#include <iostream>
#include <vector>

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

void random_samples(test_data* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? test_data::rand_host() : res[i - 1000];
}

void incremental_values(test_scalar* res, uint32_t count)
{
  for (int i = 0; i < count; i++) {
    res[i] = i ? res[i - 1] + test_scalar::one() : test_scalar::zero();
  }
}

__global__ void transpose_batch(test_scalar* in, test_scalar* out, int row_size, int column_size)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= row_size * column_size) return;
  out[(tid % row_size) * column_size + (tid / row_size)] = in[tid];
}

int main(int argc, char** argv)
{
}
