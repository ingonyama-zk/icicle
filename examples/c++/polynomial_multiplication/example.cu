#include <chrono>
#include <iostream>
#include <vector>
#include <memory>

#include "api/bn254.h"
#include "gpu-utils/error_handler.cuh"

using namespace bn254;
typedef scalar_t test_scalar;
typedef scalar_t test_data;

void random_samples(test_data* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? test_data::rand_host() : res[i - 1000];
}

void incremental_values(test_scalar* res, uint32_t count)
{
  for (int i = 0; i < count; i++) {
    res[i] = i ? res[i - 1] + test_scalar::one() * test_scalar::omega(4) : test_scalar::zero();
  }
}

// calcaulting polynomial multiplication A*B via NTT,pointwise-multiplication and INTT
// (1) allocate A,B on CPU. Randomize first half, zero second half
// (2) allocate NttAGpu, NttBGpu on GPU
// (3) calc NTT for A and for B from cpu to GPU
// (4) multiply MulGpu = NttAGpu * NttBGpu (pointwise)
// (5) INTT MulGpu inplace

int main(int argc, char** argv)
{
  cudaEvent_t start, stop;
  float measured_time;

  int NTT_LOG_SIZE = 23;
  int NTT_SIZE = 1 << NTT_LOG_SIZE;

  CHK_IF_RETURN(cudaFree(nullptr)); // init GPU context

  // init domain
  auto ntt_config = ntt::default_ntt_config<test_scalar>();
  const bool is_radix2_alg = (argc > 1) ? atoi(argv[1]) : false;
  ntt_config.ntt_algorithm = is_radix2_alg ? ntt::NttAlgorithm::Radix2 : ntt::NttAlgorithm::MixedRadix;

  const char* ntt_alg_str = is_radix2_alg ? "Radix-2" : "Mixed-Radix";
  std::cout << "Polynomial multiplication with " << ntt_alg_str << " NTT: ";

  CHK_IF_RETURN(cudaEventCreate(&start));
  CHK_IF_RETURN(cudaEventCreate(&stop));

  test_scalar basic_root = test_scalar::omega(NTT_LOG_SIZE);
  bn254_initialize_domain(&basic_root, ntt_config.ctx, true /*=fast_twidddles_mode*/);

  // (1) cpu allocation
  auto CpuA = std::make_unique<test_data[]>(NTT_SIZE);
  auto CpuB = std::make_unique<test_data[]>(NTT_SIZE);
  random_samples(CpuA.get(), NTT_SIZE >> 1); // second half zeros
  random_samples(CpuB.get(), NTT_SIZE >> 1); // second half zeros

  test_data *GpuA, *GpuB, *MulGpu;

  auto benchmark = [&](bool print, int iterations = 1) {
    // start recording
    CHK_IF_RETURN(cudaEventRecord(start, ntt_config.ctx.stream));

    for (int iter = 0; iter < iterations; ++iter) {
      // (2) gpu input allocation
      CHK_IF_RETURN(cudaMallocAsync(&GpuA, sizeof(test_data) * NTT_SIZE, ntt_config.ctx.stream));
      CHK_IF_RETURN(cudaMallocAsync(&GpuB, sizeof(test_data) * NTT_SIZE, ntt_config.ctx.stream));

      // (3) NTT for A,B from cpu to gpu
      ntt_config.are_inputs_on_device = false;
      ntt_config.are_outputs_on_device = true;
      ntt_config.ordering = ntt::Ordering::kNM;
      CHK_IF_RETURN(bn254_ntt_cuda(CpuA.get(), NTT_SIZE, ntt::NTTDir::kForward, ntt_config, GpuA));
      CHK_IF_RETURN(bn254_ntt_cuda(CpuB.get(), NTT_SIZE, ntt::NTTDir::kForward, ntt_config, GpuB));

      // (4) multiply A,B
      CHK_IF_RETURN(cudaMallocAsync(&MulGpu, sizeof(test_data) * NTT_SIZE, ntt_config.ctx.stream));
      vec_ops::VecOpsConfig config{
        ntt_config.ctx,
        true,  // is_a_on_device
        true,  // is_b_on_device
        true,  // is_result_on_device
        false  // is_async
      };
      CHK_IF_RETURN(bn254_mul_cuda(GpuA, GpuB, NTT_SIZE, config, MulGpu));

      // (5) INTT (in place)
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.ordering = ntt::Ordering::kMN;
      CHK_IF_RETURN(bn254_ntt_cuda(MulGpu, NTT_SIZE, ntt::NTTDir::kInverse, ntt_config, MulGpu));

      CHK_IF_RETURN(cudaFreeAsync(GpuA, ntt_config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(GpuB, ntt_config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(MulGpu, ntt_config.ctx.stream));
    }

    CHK_IF_RETURN(cudaEventRecord(stop, ntt_config.ctx.stream));
    CHK_IF_RETURN(cudaStreamSynchronize(ntt_config.ctx.stream));
    CHK_IF_RETURN(cudaEventElapsedTime(&measured_time, start, stop));

    if (print) { std::cout << measured_time / iterations << " MS" << std::endl; }

    return CHK_LAST();
  };

  benchmark(false); // warmup
  benchmark(true, 20);

  bn254_release_domain(ntt_config.ctx);
  CHK_IF_RETURN(cudaStreamSynchronize(ntt_config.ctx.stream));

  return 0;
}