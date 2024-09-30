#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/vec_ops.h"

#include "icicle/api/bn254.h"
using namespace bn254;

#include "examples_utils.h"

void random_samples(scalar_t* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? scalar_t::rand_host() : res[i - 1000];
}

void incremental_values(scalar_t* res, uint32_t count)
{
  for (int i = 0; i < count; i++) {
    res[i] = i ? res[i - 1] + scalar_t::one() : scalar_t::one();
  }
}

void example_element_wise(
  scalar_t* h_a, scalar_t* h_b, scalar_t* h_out, scalar_t* h_out_baseline, uint64_t N, VecOpsConfig config)
{
  std::cout << "Running example_element_wise" << std::endl;

  std::cout << std::endl << "Element-wise add" << std::endl;
  START_TIMER(baseline_add);
  for (uint64_t i = 0; i < N; i++) {
    h_out_baseline[i] = h_a[i] + h_b[i];
  }
  END_TIMER(baseline_add, "Baseline");

  START_TIMER(add);
  ICICLE_CHECK(bn254_vector_add(h_a, h_b, N, &config, h_out));
  END_TIMER(add, "  Icicle");

  for (uint64_t i = 0; i < N; i++) {
    assert(h_out[i] == h_out_baseline[i]);
  }

  std::cout << std::endl << "Element-wise multiply" << std::endl;
  START_TIMER(baseline_mul);
  for (uint64_t i = 0; i < N; i++) {
    h_out_baseline[i] = h_a[i] * h_b[i];
  }
  END_TIMER(baseline_mul, "Baseline");

  START_TIMER(mul);
  ICICLE_CHECK(bn254_vector_mul(h_a, h_b, N, &config, h_out));
  END_TIMER(mul, "  Icicle");

  for (uint64_t i = 0; i < N; i++) {
    assert(h_out[i] == h_out_baseline[i]);
  }

  std::cout << std::endl << "Element-wise divide" << std::endl;

  // START_TIMER(baseline_div);
  // for (uint64_t i = 0; i < N; i++) {
  //   h_out_baseline[i] =  h_a[i] / h_b[i];
  // }
  // END_TIMER(baseline_div, "Baseline");
  std::cout << " Baseline division not yet implemented" << std::endl;

  START_TIMER(ew_div);
  ICICLE_CHECK(vector_div(h_a, h_b, N, config, h_out));
  END_TIMER(ew_div, "  Icicle");

  for (uint64_t i = 0; i < N; i++) {
    assert(h_out[i] == h_out_baseline[i]);
  }

  return;
}

void example_scalar_vector(scalar_t* h_a, scalar_t* h_out, scalar_t* h_out_baseline, uint64_t N, VecOpsConfig config)
{
  std::cout << std::endl << "Running example_scalar_vector" << std::endl;
  std::cout << "Not implemented yet" << std::endl;
  // ICICLE_CHECK(bn254_scalar_add_vec(scalar_t::one(), h_a, N, &config, h_out));
  return;
}

void example_reduce(
  scalar_t* h_a,
  scalar_t* h_out,
  scalar_t* h_out_baseline,
  uint64_t N,
  VecOpsConfig config,
  uint64_t offset,
  uint64_t stride)
{
  std::cout << std::endl << "Running example_reduce" << std::endl;
  std::cout << std::endl << "Sum of vector elements" << std::endl;
  START_TIMER(baseline_reduce_sum);
  h_out_baseline[0] = scalar_t::zero();
  for (uint64_t i = offset; i < N; i = i + stride) {
    h_out_baseline[0] = h_out_baseline[0] + h_a[i];
  }
  END_TIMER(baseline_reduce_sum, "Baseline");

  START_TIMER(reduce_sum);
  ICICLE_CHECK(vector_sum(h_a, N, config, h_out, offset, stride));
  END_TIMER(reduce_sum, "  Icicle");

  assert(h_out[0] == h_out_baseline[0]);

  std::cout << std::endl << "Product of vector elements" << std::endl;
  START_TIMER(baseline_reduce_product);
  h_out_baseline[0] = scalar_t::one();
  for (uint64_t i = offset; i < N; i = i + stride) {
    h_out_baseline[0] = h_out_baseline[0] * h_a[i];
  }
  END_TIMER(baseline_reduce_product, "Baseline");

  START_TIMER(reduce_product);
  ICICLE_CHECK(vector_product(h_a, N, config, h_out, offset, stride));
  END_TIMER(reduce_product, "  Icicle");

  assert(h_out[0] == h_out_baseline[0]);
  return;
}

int main(int argc, char** argv)
{
  // try_load_and_set_backend_device(argc, argv);

  int N_LOG = 20;
  int N = 1 << N_LOG;
  int offset = 1;
  int stride = 4;

  // on-host data
  auto h_a = std::make_unique<scalar_t[]>(N);
  auto h_b = std::make_unique<scalar_t[]>(N);
  auto h_out = std::make_unique<scalar_t[]>(N);
  auto h_out_baseline = std::make_unique<scalar_t[]>(N);

  random_samples(h_a.get(), N);
  random_samples(h_b.get(), N);

  // incremental_values(h_a.get(), N );
  // incremental_values(h_b.get(), N );

  VecOpsConfig config = default_vec_ops_config();

  example_element_wise(h_a.get(), h_b.get(), h_out.get(), h_out_baseline.get(), N, config);
  example_reduce(h_a.get(), h_out.get(), h_out_baseline.get(), N, config, offset, stride);
  example_scalar_vector(h_a.get(), h_out.get(), h_out_baseline.get(), N, config);

  return 0;
}