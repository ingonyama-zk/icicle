#include "fields/id.h"
#define FIELD_ID M31
#define DCCT

#include "fields/field_config.cuh"
typedef field_config::scalar_t test_scalar;
typedef field_config::quad_extension_t test_ext;
typedef field_config::scalar_t test_data;

#include "fields/field.cuh"
#include "curves/projective.cuh"
#include <chrono>
#include <iostream>
#include <vector>

#include "ntt.cu"
#include "kernel_ntt.cu"
#include <memory>


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
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
  cudaEvent_t ntt_start, ntt_stop;
  float icicle_time;

  int NTT_LOG_SIZE = (argc > 1) ? atoi(argv[1]) : 6;
  int NTT_SIZE = 1 << NTT_LOG_SIZE;
  bool INPLACE = (argc > 2) ? atoi(argv[2]) : false;
  int INV = (argc > 3) ? atoi(argv[3]) : false;
  int BATCH_SIZE = (argc > 4) ? atoi(argv[4]) : 1;
  bool COLUMNS_BATCH = (argc > 5) ? atoi(argv[5]) : false;
  const ntt::Ordering ordering = (argc > 6) ? ntt::Ordering(atoi(argv[6])) : ntt::Ordering::kNR;

  // Note: NM, MN are not expected to be equal when comparing mixed-radix and radix-2 NTTs
  const char* ordering_str = ordering == ntt::Ordering::kNN   ? "NN"
                             : ordering == ntt::Ordering::kNR ? "NR"
                             : ordering == ntt::Ordering::kRN ? "RN"
                             : ordering == ntt::Ordering::kRR ? "RR"
                             : ordering == ntt::Ordering::kNM ? "NM"
                                                              : "MN";

  printf(
    "running ntt 2^%d, inplace=%d, inverse=%d, batch_size=%d, columns_batch=%d, ordering=%s\n", NTT_LOG_SIZE, INPLACE,
    INV, BATCH_SIZE, COLUMNS_BATCH, ordering_str);

  CHK_IF_RETURN(cudaFree(nullptr)); // init GPU context (warmup)

  // init domain
  auto ntt_config = ntt::default_ntt_config<test_scalar>();
  ntt_config.ordering = ordering;
  ntt_config.are_inputs_on_device = true;
  ntt_config.are_outputs_on_device = true;
  ntt_config.batch_size = BATCH_SIZE;
  ntt_config.columns_batch = COLUMNS_BATCH;

  CHK_IF_RETURN(cudaEventCreate(&ntt_start));
  CHK_IF_RETURN(cudaEventCreate(&ntt_stop));

  auto start = std::chrono::high_resolution_clock::now();
  const test_ext basic_root = field_config::get_ext_omega(NTT_LOG_SIZE);
  std::cout << "Basic root: " << basic_root << std::endl;
  ntt::init_domain<test_scalar, test_ext>(basic_root, ntt_config.ctx, false);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  std::cout << "initDomain took: " << duration / 1000 << " MS" << std::endl;

  // cpu allocation
  auto CpuScalars = std::make_unique<test_data[]>(NTT_SIZE * BATCH_SIZE);
  auto CpuOutput = std::make_unique<test_data[]>(NTT_SIZE * BATCH_SIZE);

  // gpu allocation
  scalar_t *GpuScalars, *GpuOutput;
  scalar_t* GpuScalarsTransposed;
  CHK_IF_RETURN(cudaMalloc(&GpuScalars, sizeof(test_data) * NTT_SIZE * BATCH_SIZE));
  CHK_IF_RETURN(cudaMalloc(&GpuScalarsTransposed, sizeof(test_data) * NTT_SIZE * BATCH_SIZE));
  CHK_IF_RETURN(cudaMalloc(&GpuOutput, sizeof(test_data) * NTT_SIZE * BATCH_SIZE));

  // init inputs
  incremental_values(CpuScalars.get(), NTT_SIZE * BATCH_SIZE);
  // random_samples(CpuScalars.get(), NTT_SIZE * BATCH_SIZE);
  CHK_IF_RETURN(
    cudaMemcpy(GpuScalars, CpuScalars.get(), NTT_SIZE * BATCH_SIZE * sizeof(test_data), cudaMemcpyHostToDevice));

  if (COLUMNS_BATCH) {
    transpose_batch<<<(NTT_SIZE * BATCH_SIZE + 256 - 1) / 256, 256>>>(
      GpuScalars, GpuScalarsTransposed, NTT_SIZE, BATCH_SIZE);
  }

  // CHK_IF_RETURN(
  //   ntt::ntt(GpuScalars, NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutput));
  auto iterations = 1;

  START_TIMER(ntt_timer);
  CHK_IF_RETURN(cudaEventRecord(ntt_start, ntt_config.ctx.stream));
  // OLD
  ntt_config.ntt_algorithm = ntt::NttAlgorithm::MixedRadix;
  for (size_t i = 0; i < iterations; i++) {
    CHK_IF_RETURN(
      ntt::ntt(GpuScalars, NTT_SIZE, INV ? ntt::NTTDir::kInverse : ntt::NTTDir::kForward, ntt_config, GpuOutput));
  }
  END_TIMER(ntt_timer, "NTT");
  CHK_IF_RETURN(cudaEventRecord(ntt_stop, ntt_config.ctx.stream));
  CHK_IF_RETURN(cudaStreamSynchronize(ntt_config.ctx.stream));
  CHK_IF_RETURN(cudaEventElapsedTime(&icicle_time, ntt_start, ntt_stop));

  printf("Old Runtime=%0.3f MS\n", icicle_time / (iterations));

  CHK_IF_RETURN(
    cudaMemcpy(CpuOutput.get(), GpuOutput, NTT_SIZE * BATCH_SIZE * sizeof(test_data), cudaMemcpyDeviceToHost));

  std::cout << "Output" << std::endl;
  for (int i = 0; i < NTT_SIZE * BATCH_SIZE; i++) {
    // if (i == 1024)
    //   break;
    if (i % 512 < 2)
      std::cout << CpuOutput[i] << " " << i << std::endl;
  }
  bool success = true;
  // for (int i = 0; i < NTT_SIZE * BATCH_SIZE; i++) {
  //   // if (i%64==0) printf("\n");
  //   if (CpuOutputNew[i] != CpuOutput[i]) {
  //     success = false;
  //     // std::cout << i << " ref " << CpuOutput[i] << " != " << CpuOutputNew[i] << std::endl;
  //     // break;
  //   } else {
  //     // std::cout << i << " ref " << CpuOutput[i] << " == " << CpuOutputNew[i] << std::endl;
  //     // break;
  //   }
  // }
  const char* success_str = success ? "SUCCESS!" : "FAIL!";
  // printf("%s\n", success_str);

  CHK_IF_RETURN(cudaFree(GpuScalars));
  CHK_IF_RETURN(cudaFree(GpuOutput));

  ntt::release_domain<test_scalar>(ntt_config.ctx);

  return CHK_LAST();
}