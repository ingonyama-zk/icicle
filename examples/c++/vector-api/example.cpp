#include <iostream>
#include <vector>
#include <memory>

#include "icicle/runtime.h"
#include "icicle/api/bn254.h"
#include "icicle/utils/log.h"


// SP: I undertstand this code is auto-generated, but I can't get scrip/gen to work. 

extern "C" eIcicleError bn254_vector_product(
  const bn254::scalar_t* vec_a, uint64_t n, const VecOpsConfig* config, bn254::scalar_t* result);

extern "C" eIcicleError bn254_vector_sum(
  const bn254::scalar_t* vec_a, uint64_t n, const VecOpsConfig* config, bn254::scalar_t* result);

// SP: end of my changes

using namespace bn254;

#include "examples_utils.h"

void random_samples(scalar_t* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? scalar_t::rand_host() : res[i - 1000];
}

// void incremental_values(scalar_t* res, uint32_t count)
// {
//   for (int i = 0; i < count; i++) {
//     res[i] = i ? res[i - 1] + scalar_t::one() : scalar_t::zero();
//   }
// }

int main(int argc, char** argv)
{
  try_load_and_set_backend_device(argc, argv);

  int N_LOG = 20;
  int N = 1 << N_LOG;

  // on-host data
  auto h_a = std::make_unique<scalar_t[]>(N);
  auto h_b = std::make_unique<scalar_t[]>(N);
  auto h_out = std::make_unique<scalar_t[]>(N);

  random_samples(h_a.get(), N ); 
  random_samples(h_b.get(), N ); 

  // on-device data
  scalar_t *d_a, *d_b, *d_out;

  DeviceProperties device_props;
  ICICLE_CHECK(icicle_get_device_properties(device_props));
  
  ICICLE_CHECK(icicle_malloc((void**)&d_a, sizeof(scalar_t) * N));
  ICICLE_CHECK(icicle_malloc((void**)&d_b, sizeof(scalar_t) * N));
  ICICLE_CHECK(icicle_malloc((void**)&d_out, sizeof(scalar_t) * N));

  ICICLE_CHECK(icicle_copy(d_a, h_a.get(), sizeof(scalar_t) * N)); 
  ICICLE_CHECK(icicle_copy(d_b, h_b.get(), sizeof(scalar_t) * N)); 

  VecOpsConfig h_config{
    nullptr,
    false,   // is_a_on_device
    false,   // is_b_on_device
    false,   // is_result_on_device
    false,  // is_async
    nullptr // ext
  };

  VecOpsConfig d_config{
    nullptr,
    true,   // is_a_on_device
    true,   // is_b_on_device
    true,   // is_result_on_device
    false,  // is_async
    nullptr // ext
  };


  // Reduction operations

  START_TIMER(baseline_reduce_sum);  
  h_out[0] = scalar_t::zero();
  for (uint64_t i = 0; i < N; ++i) {
    h_out[0] = h_out[0] + h_a[i];
  }
  END_TIMER(baseline_reduce_sum, "baseline reduce sum took");

  ICICLE_LOG_INFO << "Failed to load ";
  std::cout << "ext: " << std::endl;
  // d_config.ext = 2;
  std::cout << "ext: " << d_config.ext << std::endl;

  // return 0;

  START_TIMER(reduce_sum);
  ICICLE_CHECK(bn254_vector_sum(d_a, N, &d_config, d_out));
  END_TIMER(reduce_sum, "reduce sum took");


  std::cout << "h_out: " << h_out[0] << std::endl;
  std::cout << "d_out: " << d_out[0] << std::endl;




  START_TIMER(baseline_reduce_product);  
  h_out[0] = scalar_t::one();
  for (uint64_t i = 0; i < N; ++i) {
    h_out[0] = h_out[0] * h_a[i];
  }
  END_TIMER(baseline_reduce_product, "baseline reduce product took");

  
  START_TIMER(reduce_product);
  ICICLE_CHECK(bn254_vector_product(d_a, N, &d_config, d_out));
  END_TIMER(reduce_product, "reduce product took");


  std::cout << "h_out: " << h_out[0] << std::endl;
  std::cout << "d_out: " << d_out[0] << std::endl;

    

  

  ICICLE_CHECK(icicle_free(d_a));
  ICICLE_CHECK(icicle_free(d_b));
  ICICLE_CHECK(icicle_free(d_out));

  return 0;
}