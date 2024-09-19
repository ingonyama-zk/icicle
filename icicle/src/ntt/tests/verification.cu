#include "fields/id.h"
#define FIELD_ID BABY_BEAR

#ifdef ECNTT
#define CURVE_ID BN254
#include "curves/curve_config.cuh"
typedef field_config::scalar_t test_scalar;
typedef curve_config::projective_t test_data;
#else
#include "fields/field_config.cuh"
typedef field_config::scalar_t test_scalar;
typedef field_config::scalar_t test_data;
#endif

#include "fields/field.cuh"
#include "curves/projective.cuh"
#include <chrono>
#include <iostream>
#include <vector>

#include "ntt.cu"
#include "kernel_ntt.cu"
#include <memory>

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
  cudaEvent_t icicle_start, icicle_stop, new_start, new_stop;
  float icicle_time, new_time;

  int NTT_LOG_SIZE = (argc > 1) ? atoi(argv[1]) : 19;
  int NTT_SIZE = 1 << NTT_LOG_SIZE;
  bool INPLACE = (argc > 2) ? atoi(argv[2]) : false;
  int INV = (argc > 3) ? atoi(argv[3]) : false;
  int BATCH_SIZE = (argc > 4) ? atoi(argv[4]) : 150;
  bool COLUMNS_BATCH = (argc > 5) ? atoi(argv[5]) : false;
  int COSET_IDX = (argc > 6) ? atoi(argv[6]) : 2;
  const ntt::Ordering ordering = (argc > 7) ? ntt::Ordering(atoi(argv[7])) : ntt::Ordering::kNN;
  bool FAST_TW = (argc > 8) ? atoi(argv[8]) : true;

  // Note: NM, MN are not expected to be equal when comparing mixed-radix and radix-2 NTTs
  const char* ordering_str = ordering == ntt::Ordering::kNN   ? "NN"
                             : ordering == ntt::Ordering::kNR ? "NR"
                             : ordering == ntt::Ordering::kRN ? "RN"
                             : ordering == ntt::Ordering::kRR ? "RR"
                             : ordering == ntt::Ordering::kNM ? "NM"
                                                              : "MN";

  printf(
    "running ntt 2^%d, inplace=%d, inverse=%d, batch_size=%d, columns_batch=%d coset-idx=%d, ordering=%s, fast_tw=%d\n",
    NTT_LOG_SIZE, INPLACE, INV, BATCH_SIZE, COLUMNS_BATCH, COSET_IDX, ordering_str, FAST_TW);

  CHK_IF_RETURN(cudaFree(nullptr)); // init GPU context (warmup)

  // init domain
  auto ntt_config = ntt::default_ntt_config<test_scalar>();
  ntt_config.ordering = ordering;
  ntt_config.are_inputs_on_device = true;
  ntt_config.are_outputs_on_device = true;
  ntt_config.batch_size = BATCH_SIZE;
  ntt_config.columns_batch = COLUMNS_BATCH;

  CHK_IF_RETURN(cudaEventCreate(&icicle_start));
  CHK_IF_RETURN(cudaEventCreate(&icicle_stop));
  CHK_IF_RETURN(cudaEventCreate(&new_start));
  CHK_IF_RETURN(cudaEventCreate(&new_stop));

  auto start = std::chrono::high_resolution_clock::now();
  const scalar_t basic_root = test_scalar::omega(NTT_LOG_SIZE);
  ntt::init_domain<scalar_t, scalar_t>(basic_root, ntt_config.ctx, FAST_TW);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  std::cout << "initDomain took: " << duration / 1000 << " MS" << std::endl;

  // cpu allocation
  auto CpuScalars = std::make_unique<test_data[]>(NTT_SIZE * BATCH_SIZE);
  auto CpuOutputOld = std::make_unique<test_data[]>(NTT_SIZE * BATCH_SIZE);
  auto CpuOutputNew = std::make_unique<test_data[]>(NTT_SIZE * BATCH_SIZE);

  // gpu allocation
  scalar_t *GpuScalars, *GpuOutputOld, *GpuOutputNew;
  scalar_t* GpuScalarsTransposed;
  CHK_IF_RETURN(cudaMalloc(&GpuScalars, sizeof(test_data) * NTT_SIZE * BATCH_SIZE));
  CHK_IF_RETURN(cudaMalloc(&GpuScalarsTransposed, sizeof(test_data) * NTT_SIZE * BATCH_SIZE));
  CHK_IF_RETURN(cudaMalloc(&GpuOutputOld, sizeof(test_data) * NTT_SIZE * BATCH_SIZE));
  CHK_IF_RETURN(cudaMalloc(&GpuOutputNew, sizeof(test_data) * NTT_SIZE * BATCH_SIZE));

  // init inputs
  // incremental_values(CpuScalars.get(), NTT_SIZE * BATCH_SIZE);
  random_samples(CpuScalars.get(), NTT_SIZE * BATCH_SIZE);
  CHK_IF_RETURN(
    cudaMemcpy(GpuScalars, CpuScalars.get(), NTT_SIZE * BATCH_SIZE * sizeof(test_data), cudaMemcpyHostToDevice));

  if (COLUMNS_BATCH) {
    transpose_batch<<<(NTT_SIZE * BATCH_SIZE + 256 - 1) / 256, 256>>>(
      GpuScalars, GpuScalarsTransposed, NTT_SIZE, BATCH_SIZE);
  }

  // inplace
  if (INPLACE) {
    CHK_IF_RETURN(cudaMemcpy(
      GpuOutputNew, COLUMNS_BATCH ? GpuScalarsTransposed : GpuScalars, NTT_SIZE * BATCH_SIZE * sizeof(test_data),
      cudaMemcpyDeviceToDevice));
  }

  for (int coset_idx = 0; coset_idx < COSET_IDX; ++coset_idx) {
    ntt_config.coset_gen = ntt_config.coset_gen * basic_root;
  }

  auto benchmark = [&](bool is_print, int iterations) -> cudaError_t {
    // NEW
    CHK_IF_RETURN(cudaEventRecord(new_start, ntt_config.ctx.stream));
    ntt_config.ntt_algorithm = ntt::NttAlgorithm::MixedRadix;
    for (size_t i = 0; i < iterations; i++) {
      CHK_IF_RETURN(ntt::ntt(
        INPLACE         ? GpuOutputNew
        : COLUMNS_BATCH ? GpuScalarsTransposed
                        : GpuScalars,
        NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutputNew));
    }
    CHK_IF_RETURN(cudaEventRecord(new_stop, ntt_config.ctx.stream));
    CHK_IF_RETURN(cudaStreamSynchronize(ntt_config.ctx.stream));
    CHK_IF_RETURN(cudaEventElapsedTime(&new_time, new_start, new_stop));

    // OLD
    CHK_IF_RETURN(cudaEventRecord(icicle_start, ntt_config.ctx.stream));
    ntt_config.ntt_algorithm = ntt::NttAlgorithm::Radix2;
    for (size_t i = 0; i < iterations; i++) {
      CHK_IF_RETURN(
        ntt::ntt(GpuScalars, NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutputOld));
    }
    CHK_IF_RETURN(cudaEventRecord(icicle_stop, ntt_config.ctx.stream));
    CHK_IF_RETURN(cudaStreamSynchronize(ntt_config.ctx.stream));
    CHK_IF_RETURN(cudaEventElapsedTime(&icicle_time, icicle_start, icicle_stop));

    if (is_print) {
      printf("Old Runtime=%0.3f MS\n", icicle_time / iterations);
      printf("New Runtime=%0.3f MS\n", new_time / iterations);
    }

    return CHK_LAST();
  };

  CHK_IF_RETURN(benchmark(false /*=print*/, 1)); // warmup
  int count = INPLACE ? 1 : 10;
  if (INPLACE) {
    CHK_IF_RETURN(cudaMemcpy(
      GpuOutputNew, COLUMNS_BATCH ? GpuScalarsTransposed : GpuScalars, NTT_SIZE * BATCH_SIZE * sizeof(test_data),
      cudaMemcpyDeviceToDevice));
  }
  CHK_IF_RETURN(benchmark(true /*=print*/, count));

  if (COLUMNS_BATCH) {
    transpose_batch<<<(NTT_SIZE * BATCH_SIZE + 256 - 1) / 256, 256>>>(
      GpuOutputNew, GpuScalarsTransposed, BATCH_SIZE, NTT_SIZE);
    CHK_IF_RETURN(cudaMemcpy(
      GpuOutputNew, GpuScalarsTransposed, NTT_SIZE * BATCH_SIZE * sizeof(test_data), cudaMemcpyDeviceToDevice));
  }

  // verify
  CHK_IF_RETURN(
    cudaMemcpy(CpuOutputNew.get(), GpuOutputNew, NTT_SIZE * BATCH_SIZE * sizeof(test_data), cudaMemcpyDeviceToHost));
  CHK_IF_RETURN(
    cudaMemcpy(CpuOutputOld.get(), GpuOutputOld, NTT_SIZE * BATCH_SIZE * sizeof(test_data), cudaMemcpyDeviceToHost));

  bool success = true;
  for (int i = 0; i < NTT_SIZE * BATCH_SIZE; i++) {
    // if (i%64==0) printf("\n");
    if (CpuOutputNew[i] != CpuOutputOld[i]) {
      success = false;
      // std::cout << i << " ref " << CpuOutputOld[i] << " != " << CpuOutputNew[i] << std::endl;
      // break;
    } else {
      // std::cout << i << " ref " << CpuOutputOld[i] << " == " << CpuOutputNew[i] << std::endl;
      // break;
    }
  }
  const char* success_str = success ? "SUCCESS!" : "FAIL!";
  printf("%s\n", success_str);

  CHK_IF_RETURN(cudaFree(GpuScalars));
  CHK_IF_RETURN(cudaFree(GpuOutputOld));
  CHK_IF_RETURN(cudaFree(GpuOutputNew));

  ntt::release_domain<test_scalar>(ntt_config.ctx);

  return CHK_LAST();
}