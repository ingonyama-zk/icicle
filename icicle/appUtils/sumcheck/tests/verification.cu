#define CURVE_ID BLS12_381

#include "primitives/field.cuh"
#include "primitives/projective.cuh"
#include <chrono>
#include <iostream>
#include <vector>

#include "curves/curve_config.cuh"
#include "sumcheck/sumcheck.cu"
#include <memory>

typedef curve_config::scalar_t test_scalar;

void random_samples(test_scalar* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? test_scalar::rand_host() : res[i - 1000];
}

void incremental_values(test_scalar* res, uint32_t count)
{
  for (int i = 0; i < count; i++) {
    res[i] = i ? res[i - 1] + test_scalar::one() : test_scalar::zero();
  }
}

int main(){

  //decleration
  test_scalar *d_transcript;
  test_scalar *d_evals;
  test_scalar *d_temp;
  test_scalar *d_transcript2;
  test_scalar *d_evals2;
  test_scalar *d_temp2;
  test_scalar C;

  cudaEvent_t gpu_start, gpu_stop;
  float gpu_time;

  int n = 16;
  int size = 1 << n;

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  //allocation
  auto h_evals = std::make_unique<test_scalar[]>(size);
  auto h_temp = std::make_unique<test_scalar[]>(size);
  auto h_transcript = std::make_unique<test_scalar[]>(2*n+1);
  auto h_transcript_ref = std::make_unique<test_scalar[]>(2*n+1);
  
  cudaMalloc(&d_transcript, sizeof(test_scalar) * (2*n+1));
  cudaMalloc(&d_evals, sizeof(test_scalar) * size);
  cudaMalloc(&d_temp, sizeof(test_scalar) * size);
  cudaMalloc(&d_transcript2, sizeof(test_scalar) * (2*n+1));
  cudaMalloc(&d_evals2, sizeof(test_scalar) * size);
  cudaMalloc(&d_temp2, sizeof(test_scalar) * size);
  
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);

  //input init
  // random_samples(h_evals.get(), size);
  incremental_values(h_evals.get(), size);
  C = test_scalar::rand_host();
  h_transcript[0] = test_scalar::rand_host();
  h_transcript_ref[0] = h_transcript[0];

  //warm up run
  cudaMemcpy(d_evals, h_evals.get(), sizeof(test_scalar) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_transcript, h_transcript.get(), sizeof(test_scalar), cudaMemcpyHostToDevice);
  // sumcheck_alg1(d_evals, d_temp, d_transcript, C, n, stream1);
  sumcheck_alg1_unified(d_evals, d_temp, d_transcript, C, n, stream1);
  // sumcheck_alg1(d_evals2, d_temp2, d_transcript2, C, n, stream2);
  cudaDeviceSynchronize();
  cudaMemcpy(d_evals, h_evals.get(), sizeof(test_scalar) * size, cudaMemcpyHostToDevice);

  //run
  cudaEventRecord(gpu_start, 0);
  // sumcheck_alg1(d_evals, d_temp, d_transcript, C, n, stream1);
  sumcheck_alg1_unified(d_evals, d_temp, d_transcript, C, n, stream1);
  // sumcheck_alg1(d_evals2, d_temp2, d_transcript2, C, n, stream2);
  cudaEventRecord(gpu_stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  //run reference
  auto cpu_start = std::chrono::high_resolution_clock::now();
  sumcheck_alg1_ref(h_evals.get(), h_temp.get(), h_transcript_ref.get(), C, n);
  auto cpu_stop = std::chrono::high_resolution_clock::now();
  auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();

  //verify
  cudaMemcpy(h_transcript.get(), d_transcript, sizeof(test_scalar) * (2*n+1), cudaMemcpyDeviceToHost);
  
  bool success = true;
  for (int i = 0; i < 2*n+1; i++) {
    if (h_transcript[i] != h_transcript_ref[i]) {
      success = false;
      std::cout << i << " ref " << h_transcript_ref[i] << " != " << h_transcript[i] << std::endl;
    } else {
      std::cout << i << " ref " << h_transcript_ref[i] << " == " << h_transcript[i] << std::endl;
    }
  }
  const char* success_str = success ? "SUCCESS!" : "FAIL!";
  printf("%s\n", success_str);
  
  //print times
  printf("GPU Runtime=%0.3f MS\n", gpu_time);
  std::cout << "CPU Runtime=" << cpu_time / 1000 << " MS" << std::endl;
  // printf("CPU Runtime=%0.3f MS\n", cpu_time);

  //free
  cudaFree(d_evals);
  cudaFree(d_temp);
  cudaFree(d_transcript);


}


// int main(){

//   int evals[8] = {0,1,2,3,4,5,6,7};
//   int t[8] = {0,0,0,0,0,0,0,0};
//   int T[8] = {45,0,0,0,0,0,0,0};
//   int C = 33;
//   int n = 3;

//   int *d_evals;
//   int *d_t;
//   int *d_T;
  
//   int log_size = 17;
//   int size = 1<<log_size;
//   auto largEvals = std::make_unique<int[]>(size);
//   for (int i = 0; i < size; i++)
//   {
//     largEvals[i] = i%2? 1 : 2;
//   }
  

//   // cudaMalloc(&d_evals, sizeof(int) * size);
//   // cudaMemcpy(d_evals, largEvals.get(), sizeof(int) * size, cudaMemcpyHostToDevice);
//   cudaMalloc(&d_evals, sizeof(int) * 8);
//   cudaMemcpy(d_evals, evals, sizeof(int) * 8, cudaMemcpyHostToDevice);
//   cudaMalloc(&d_t, sizeof(int) * 8);
//   cudaMalloc(&d_T, sizeof(int) * 8);
//   cudaMemcpy(d_t, t, sizeof(int) * 8, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_T, T, sizeof(int) * 8, cudaMemcpyHostToDevice);

//   for (int i = 0; i < 8; i++)
//   {
//     std::cout << evals[i] <<std::endl;
//   }
//   // accumulate(d_evals, d_evals, log_size);
//   // cudaDeviceSynchronize();
//   // printf("cuda err %d\n", cudaGetLastError());
//   // cudaMemcpy(largEvals.get(), d_evals, sizeof(int) * size, cudaMemcpyDeviceToHost);
//   // for (int i = 0; i < 8; i++)
//   // {
//   //   std::cout << largEvals[i] <<std::endl;
//   // }


//   sumcheck_alg1(d_evals, d_t, d_T, C, n);
//   // sumcheck_alg1_ref(evals, t, T, C, n);

//   // for (int i = 0; i < 8; i++)
//   // {
//   //   std::cout << T[i] <<std::endl;
//   // }

//   cudaMemcpy(T, d_T, sizeof(int) * 8, cudaMemcpyDeviceToHost);

//   for (int i = 0; i < 8; i++)
//   {
//     std::cout << T[i] <<std::endl;
//   }

//   return 0;
// }