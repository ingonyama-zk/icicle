
#define CURVE_ID 1 // TODO Yuval: move to makefile

#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/cuda_utils.cuh"
#include <chrono>
#include <iostream>
#include <vector>

#include "curves/curve_config.cuh"
#include "ntt/ntt.cu"
#include "large_ntt/large_ntt.cuh"
#include <memory>

#define PERFORMANCE

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

  int NTT_LOG_SIZE = 21;
  int NTT_SIZE = 1 << NTT_LOG_SIZE;
  int INV = false;
  const ntt::Ordering ordering = ntt::Ordering::kNN;
  const char* ordering_str = ordering == ntt::Ordering::kNN   ? "NN"
                             : ordering == ntt::Ordering::kNR ? "NR"
                             : ordering == ntt::Ordering::kRN ? "RN"
                                                              : "RR";

  printf("running ntt 2^%d, INV=%d, ordering=%s\n", NTT_LOG_SIZE, INV, ordering_str);

  // cpu allocation
  auto CpuScalars = std::make_unique<test_scalar[]>(NTT_SIZE);
  auto CpuOutputOld = std::make_unique<test_scalar[]>(NTT_SIZE);
  auto CpuOutputNew = std::make_unique<test_scalar[]>(NTT_SIZE);

  // gpu allocation
  test_scalar *GpuScalars, *GpuOutputOld, *GpuOutputNew;
  $CUDA(cudaMalloc(&GpuScalars, sizeof(test_scalar) * NTT_SIZE));
  $CUDA(cudaMalloc(&GpuOutputOld, sizeof(test_scalar) * NTT_SIZE));
  $CUDA(cudaMalloc(&GpuOutputNew, sizeof(test_scalar) * NTT_SIZE));

  // init inputs
  random_samples(CpuScalars.get(), NTT_SIZE);
  $CUDA(cudaMemcpy(GpuScalars, CpuScalars.get(), NTT_SIZE, cudaMemcpyHostToDevice));

  // new algorithm init
  // ntt::MixedRadixNTT new_ntt(NTT_SIZE, INV, ordering);
  // old algorithm init
  auto ntt_config = ntt::DefaultNTTConfig<test_scalar>();
  ntt_config.ordering = ordering;
  ntt_config.are_inputs_on_device = true;
  ntt_config.are_outputs_on_device = true;
  ntt_config.is_force_radix2 = true; // to compare to radix2 algorithm
  const test_scalar basic_root = test_scalar::omega(NTT_LOG_SIZE);
  ntt::InitDomain(basic_root, ntt_config.ctx);

#ifdef PERFORMANCE
  $CUDA(cudaEventCreate(&icicle_start));
  $CUDA(cudaEventCreate(&icicle_stop));
  $CUDA(cudaEventCreate(&new_start));
  $CUDA(cudaEventCreate(&new_stop));

  // run ntt
  auto benchmark = [&](bool is_print, int iterations) {
    $CUDA(cudaEventRecord(new_start, 0));
    for (size_t i = 0; i < iterations; i++) {
      // Note: measuring construction/destruction everytime since this is what real usecase is doing
      ntt::MixedRadixNTT new_ntt(NTT_SIZE, INV, ordering);
      new_ntt(GpuScalars, GpuOutputNew);
    }
    $CUDA(cudaEventRecord(new_stop, 0));
    $CUDA(cudaDeviceSynchronize());
    $CUDA(cudaEventElapsedTime(&new_time, new_start, new_stop));
    cudaDeviceSynchronize();
    if (is_print) { printf("cuda err %d\n", cudaGetLastError()); }

    $CUDA(cudaEventRecord(icicle_start, 0));
    for (size_t i = 0; i < iterations; i++) {
      ntt::NTT(GpuScalars, NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutputOld);
    }
    $CUDA(cudaEventRecord(icicle_stop, 0));
    $CUDA(cudaDeviceSynchronize());
    $CUDA(cudaEventElapsedTime(&icicle_time, icicle_start, icicle_stop));
    cudaDeviceSynchronize();
    if (is_print) {
      printf("cuda err %d\n", cudaGetLastError());
      fprintf(stderr, "Old Runtime=%0.3f MS\n", icicle_time / iterations);
      fprintf(stderr, "New Runtime=%0.3f MS\n", new_time / iterations);
    }
  };

  int count = 1;
  benchmark(false /*=print*/, 1); // warmup - is this applicable to real usecase??
  benchmark(true /*=print*/, count);
#else
  new_ntt(GpuScalars, GpuOutputNew);
  cudaDeviceSynchronize();
  printf("finished new\n");

  ntt::NTT(GpuScalars, NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutputOld);
  printf("finished old\n");

  // verify
  $CUDA(cudaMemcpy(CpuOutputNew.get(), GpuOutputNew, NTT_SIZE, cudaMemcpyDeviceToHost));
  $CUDA(cudaMemcpy(CpuOutputOld.get(), GpuOutputOld, NTT_SIZE, cudaMemcpyDeviceToHost));

  bool success = true;
  for (int i = 0; i < NTT_SIZE; i++) {
    if (CpuOutputNew[i] != CpuOutputOld[i]) {
      success = false;
      std::cout << i << " ref " << CpuOutputOld[i] << " != " << CpuOutputNew[i] << std::endl;
      // break;
    } else {
      // std::cout << i << " ref " << CpuOutputOld[i] << " == " << CpuOutputNew[i] << std::endl;
      // break;
    }
  }
  const char* success_str = success ? "SUCCESS!" : "FAIL!";
  printf("%s\n", success_str);
#endif

  $CUDA(cudaFree(GpuScalars));
  $CUDA(cudaFree(GpuOutputOld));
  $CUDA(cudaFree(GpuOutputNew));

  return 0;
}