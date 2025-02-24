#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

#include "icicle/runtime.h"
#include "icicle/api/babybear.h"
#include "icicle/utils/log.h"


// SP: I undertstand this code is auto-generated, but I can't get scrip/gen_c_api.py to work. 

extern "C" eIcicleError babybear_vector_product(
  const babybear::scalar_t* vec_a, uint64_t n, const VecOpsConfig* config, babybear::scalar_t* result, uint64_t offset, uint64_t  stride);

extern "C" eIcicleError babybear_vector_sum(
  const babybear::scalar_t* vec_a, uint64_t n, const VecOpsConfig* config, babybear::scalar_t* result, uint64_t offset, uint64_t  stride);

extern "C" eIcicleError babybear_vector_div(const babybear::scalar_t* vec_a, const babybear::scalar_t* vec_b, uint64_t size, const VecOpsConfig* config, babybear::scalar_t* output);

// SP: end of my changes
using namespace icicle;
using namespace babybear;

#include "examples_utils.h"

using FpMiliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define MY_START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define MY_END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf("%s: %.3f ms\n", msg, FpMiliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());
#define MY_END_TIMER_AVERAGE(timer, msg, enable, iters)                                                                   \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMiliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / iters);


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

void vector_equal(scalar_t* a, scalar_t* b, uint64_t N)
{
    int  count = 0;
    for (uint64_t i = 0; i < N; i++) {
        if(a[i] != b[i]) {
            count++;
            if (count < 10) {
                ICICLE_LOG_ERROR << "Mismatch at index " << i;
                ICICLE_LOG_ERROR << "a[" << i << "] = " << a[i];
                ICICLE_LOG_ERROR << "b[" << i << "] = " << b[i];
            }
        }
    }
    if(count > 0) {
        ICICLE_LOG_INFO << "Vectors are not equal: " << count << " mismatches";
    } else {
        ICICLE_LOG_INFO << "Vectors are equal";
    }
}

void example_element_wise(
  scalar_t* h_a,
  scalar_t* h_b, 
  scalar_t* h_out, 
  scalar_t* h_out_baseline, 
  uint64_t N, 
  VecOpsConfig config) 
{
  std::cout << std::endl << "Running example_element_wise" << std::endl;
  //    ICICLE_LOG_INFO << "Element-wise operations on vectors of size " << N;
  int iter = 1;
  std::cout << "Element-wise add" << std::endl; 
  START_TIMER(baseline_add);  
  for (uint64_t i = 0; i < N; i++) {
  h_out_baseline[i] =  h_a[i] + h_b[i];
  }
  END_TIMER(baseline_add, "Baseline"); 

  // START_TIMER(add);
  babybear_vector_add(h_a, h_b, N, &config, h_out);
  // END_TIMER(add, "  Icicle");

  //  vector_equal(h_out, h_out_baseline, N);
    
  // std::cout << std::endl << "Element-wise multiply" << std::endl;
  // MY_START_TIMER(baseline_mul);  
  // for (uint64_t i = 0; i < N; i++) {
  //   h_out_baseline[i] =  h_a[i] * h_b[i];
  // }
  // MY_END_TIMER_AVERAGE(baseline_mul, "Baseline", true, 1);

  // MY_START_TIMER(mul);
  // for (int i = 0; i < iter; i++) {
  // ICICLE_CHECK(babybear_vector_mul(h_a, h_b, N, &config, h_out));
  // }
  // MY_END_TIMER_AVERAGE(mul, "  Icicle", true, iter);
//
//  for (uint64_t i = 0; i < N; i++) {
//    assert(h_out[i] == h_out_baseline[i]);
//  }
//
//  std::cout << std::endl << "Element-wise divide" << std::endl;

  // START_TIMER(baseline_div);  
  // for (uint64_t i = 0; i < N; i++) {
  //   h_out_baseline[i] =  h_a[i] / h_b[i];
  // }
  // END_TIMER(baseline_div, "Baseline");
//  std::cout << " Baseline division not yet implemented" << std::endl;

//  START_TIMER(ew_div);
//  ICICLE_CHECK(babybear_vector_div(h_a, h_b, N, &config, h_out));
//  END_TIMER(ew_div, "  Icicle");

//  for (uint64_t i = 0; i < N; i++) {
//    assert(h_out[i] == h_out_baseline[i]);
//  }

  return;
}

void example_scalar_vector(
  scalar_t* h_a, 
  scalar_t* h_out, 
  scalar_t* h_out_baseline, 
  uint64_t N, 
  VecOpsConfig config) 
{
  std::cout << std::endl << "Running example_scalar_vector" << std::endl;
  std::cout << "Not implemented yet" << std::endl;
  // ICICLE_CHECK(babybear_scalar_add_vec(scalar_t::one(), h_a, N, &config, h_out));
  return;
}

// void example_reduce(
//   scalar_t* h_a, 
//   scalar_t* h_out, 
//   scalar_t* h_out_baseline, 
//   uint64_t N, 
//   VecOpsConfig config,
//   uint64_t offset, 
//   uint64_t stride) 
// {
//   std::cout << std::endl << "Running example_reduce" << std::endl;
//   std::cout << std::endl << "Sum of vector elements" << std::endl;
//   START_TIMER(baseline_reduce_sum);  
//   h_out_baseline[0] = scalar_t::zero();
//   for (uint64_t i = offset; i < N; i=i+stride) {
//     h_out_baseline[0] = h_out_baseline[0] + h_a[i];
//   }
//   END_TIMER(baseline_reduce_sum, "Baseline");

//   START_TIMER(reduce_sum);
//   ICICLE_CHECK(babybear_vector_sum(h_a, N, &config, h_out, offset, stride));
//   END_TIMER(reduce_sum, "  Icicle");

//   assert(h_out[0] == h_out_baseline[0]);

//   std::cout << std::endl << "Product of vector elements" << std::endl;
//   START_TIMER(baseline_reduce_product);  
//   h_out_baseline[0] = scalar_t::one();
//   for (uint64_t i = offset; i < N; i = i + stride) {
//     h_out_baseline[0] = h_out_baseline[0] * h_a[i];
//   }
//   END_TIMER(baseline_reduce_product, "Baseline");
  
//   START_TIMER(reduce_product);
//   ICICLE_CHECK(babybear_vector_product(h_a, N, &config, h_out, offset, stride));
//   END_TIMER(reduce_product, "  Icicle");

//   assert(h_out[0] == h_out_baseline[0]);
//   return;
// }


int main(int argc, char** argv)
{
  Log::set_min_log_level(Log::eLogLevel::Verbose);
  try_load_and_set_backend_device(argc, argv);

  


  int N_LOG = 10;
  int N = 1 << N_LOG;
  int offset = 1;
  int stride = 4;


  scalar_t *dev_a, *dev_b, *dev_c;
  std::cout << "SP: Allocating memory on device" << std::endl;
  ICICLE_CHECK(icicle_malloc((void**)&dev_a, N * sizeof(scalar_t)));
  ICICLE_CHECK(icicle_malloc((void**)&dev_b, N * sizeof(scalar_t)));
  ICICLE_CHECK(icicle_malloc((void**)&dev_c, N * sizeof(scalar_t)));
  std::cout << "Checked in-device malloc" << std::endl;
  auto config_ondevice = default_vec_ops_config();
  config_ondevice.is_a_on_device = true;
  config_ondevice.is_b_on_device = true;
  config_ondevice.is_result_on_device = true;

  babybear_vector_add(dev_a, dev_b, N, &config_ondevice, dev_c);

  // free memory on device
  ICICLE_CHECK(icicle_free(dev_a));
  ICICLE_CHECK(icicle_free(dev_b));
  ICICLE_CHECK(icicle_free(dev_c));
  return 0;
  
  
  // // on-host data
  // auto h_a = std::make_unique<scalar_t[]>(N);
  // auto h_b = std::make_unique<scalar_t[]>(N);
  // auto h_out = std::make_unique<scalar_t[]>(N);
  // auto h_out_baseline = std::make_unique<scalar_t[]>(N);

  // random_samples(h_a.get(), N ); 
  // random_samples(h_b.get(), N ); 

  // auto config = default_vec_ops_config();
  // std::cout << "*** Warm-up ***" << std::endl;
  // example_element_wise(h_a.get(), h_b.get(), h_out.get(), h_out_baseline.get(), N, config);
  // std::cout << "***********************" << std::endl;
  // std::cout << "*** Benchmark run 1 ***" << std::endl;
  // std::cout << "***********************" << std::endl;
  // example_element_wise(h_a.get(), h_b.get(), h_out.get(), h_out_baseline.get(), N, config);
  // std::cout << "***********************" << std::endl;
  // std::cout << "*** Benchmark run 2 ***" << std::endl;
  // std::cout << "***********************" << std::endl;
  // example_element_wise(h_a.get(), h_b.get(), h_out.get(), h_out_baseline.get(), N, config);
  // // example_reduce(h_a.get(), h_out.get(), h_out_baseline.get(), N, config, offset, stride);
  // // example_scalar_vector(h_a.get(), h_out.get(), h_out_baseline.get(), N, config);
  
  

  return 0;
}
