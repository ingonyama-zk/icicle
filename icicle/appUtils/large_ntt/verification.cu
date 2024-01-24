
#define CURVE_ID 3 // TODO Yuval: move to makefile

#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/cuda_utils.cuh"
#include <chrono>
#include <iostream>
#include <vector>

#include "curves/curve_config.cuh"
#include "ntt/ntt.cu"
#include "large_ntt/large_ntt.cuh"

// #define PERFORMANCE

typedef curve_config::scalar_t test_scalar;
#include "kernel_ntt.cu"

#define $CUDA(call)                                                                                                    \
  if ((call) != 0) {                                                                                                   \
    printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, cudaGetLastError());        \
    exit(1);                                                                                                           \
  }

void random_samples(test_scalar* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? test_scalar::rand_host() : res[i - 1000];
}

void incremental_values(test_scalar* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i ? res[i - 1] + test_scalar::one() * test_scalar::omega(4) : test_scalar::zero();
}

int main()
{
#ifdef PERFORMANCE
  cudaEvent_t icicle_start, icicle_stop, new_start, new_stop;
  float icicle_time, new_time;
#endif

  int NTT_LOG_SIZE = 19;
  int NTT_SIZE = 1 << NTT_LOG_SIZE;
  int INV = false;
  const ntt::Ordering ordering = ntt::Ordering::kNN;
  const char* ordering_str = ordering == ntt::Ordering::kNN   ? "NN"
                             : ordering == ntt::Ordering::kNR ? "NR"
                             : ordering == ntt::Ordering::kRN ? "RN"
                                                              : "RR";

  printf("running ntt 2^%d, INV=%d, ordering=%s\n", NTT_LOG_SIZE, INV, ordering_str);

  // cpu allocation
  test_scalar* cpuScalars = (test_scalar*)malloc(sizeof(test_scalar) * NTT_SIZE);
  if (cpuScalars == NULL) {
    fprintf(stderr, "Malloc failed\n");
    exit(1);
  }

  // gpu allocation
  test_scalar *gpuScalars, *GpuOutputOld, *GpuOutputNew;
  $CUDA(cudaMallocManaged((void**)&gpuScalars, sizeof(test_scalar) * NTT_SIZE));
  $CUDA(cudaMallocManaged((void**)&GpuOutputOld, sizeof(test_scalar) * NTT_SIZE));
  $CUDA(cudaMallocManaged((void**)&GpuOutputNew, sizeof(test_scalar) * NTT_SIZE));

  // init inputs
  random_samples(gpuScalars, NTT_SIZE);

  // new algorithm init
  ntt::MixedRadixNTT new_ntt(NTT_SIZE, INV, ordering);
  const test_scalar basic_root = INV ? test_scalar::omega_inv(NTT_LOG_SIZE) : test_scalar::omega(NTT_LOG_SIZE);
  // old algorithm init
  auto ntt_config = ntt::DefaultNTTConfig<test_scalar>();
  ntt_config.ordering = ordering;
  ntt_config.are_inputs_on_device = true;
  ntt_config.are_outputs_on_device = true;
  ntt_config.is_force_radix2 = true; // to compare to radix2 algorithm
  ntt::InitDomain(basic_root, ntt_config.ctx);

#ifdef PERFORMANCE
  $CUDA(cudaEventCreate(&icicle_start));
  $CUDA(cudaEventCreate(&icicle_stop));
  $CUDA(cudaEventCreate(&new_start));
  $CUDA(cudaEventCreate(&new_stop));

  // run ntts
  int count = 100;
  $CUDA(cudaEventRecord(new_start, 0));
  for (size_t i = 0; i < count; i++) {
    new_ntt(gpuScalars, GpuOutputNew);
  }
  $CUDA(cudaEventRecord(new_stop, 0));
  $CUDA(cudaDeviceSynchronize());
  $CUDA(cudaEventElapsedTime(&new_time, new_start, new_stop));
  cudaDeviceSynchronize();
  printf("cuda err %d\n", cudaGetLastError());

  $CUDA(cudaEventRecord(icicle_start, 0));
  for (size_t i = 0; i < count; i++) {
    ntt::NTT(gpuScalars, NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutputOld);
  }
  $CUDA(cudaEventRecord(icicle_stop, 0));
  $CUDA(cudaDeviceSynchronize());
  $CUDA(cudaEventElapsedTime(&icicle_time, icicle_start, icicle_stop));
  cudaDeviceSynchronize();
  printf("cuda err %d\n", cudaGetLastError());
  fprintf(stderr, "Old Runtime=%0.3f MS\n", icicle_time / count);
  fprintf(stderr, "New Runtime=%0.3f MS\n", new_time / count);
#else
  new_ntt(gpuScalars, GpuOutputNew);
  printf("finished new\n");

  ntt::NTT(gpuScalars, NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutputOld);
  printf("finished old\n");

  // verify
  bool success = true;
  for (int i = 0; i < NTT_SIZE; i++) {
    if (GpuOutputNew[i] != GpuOutputOld[i]) {
      success = false;
      std::cout << i << " ref " << GpuOutputOld[i] << " != " << GpuOutputNew[i] << std::endl;
      // break;
    } else {
      // std::cout << i << " ref " << icicle_temp << " == " << new_temp << std::endl;
    }
  }
  const char* success_str = success ? "SUCCESS!" : "FAIL!";
  printf("%s\n", success_str);
#endif

  return 0;
}