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
typedef curve_config::scalar_t test_data;


int main(){

  int evals[8] = {0,1,2,3,4,5,6,7};
  int t[8] = {0,0,0,0,0,0,0,0};
  int T[8] = {45,0,0,0,0,0,0,0};
  int C = 33;
  int n = 3;

  int *d_evals;
  int *d_t;
  int *d_T;
  
  int log_size = 17;
  int size = 1<<log_size;
  auto largEvals = std::make_unique<int[]>(size);
  for (int i = 0; i < size; i++)
  {
    largEvals[i] = i%2? 1 : 2;
  }
  

  // cudaMalloc(&d_evals, sizeof(int) * size);
  // cudaMemcpy(d_evals, largEvals.get(), sizeof(int) * size, cudaMemcpyHostToDevice);
  cudaMalloc(&d_evals, sizeof(int) * 8);
  cudaMemcpy(d_evals, evals, sizeof(int) * 8, cudaMemcpyHostToDevice);
  cudaMalloc(&d_t, sizeof(int) * 8);
  cudaMalloc(&d_T, sizeof(int) * 8);
  cudaMemcpy(d_t, t, sizeof(int) * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(d_T, T, sizeof(int) * 8, cudaMemcpyHostToDevice);

  for (int i = 0; i < 8; i++)
  {
    std::cout << evals[i] <<std::endl;
  }
  // accumulate(d_evals, d_evals, log_size);
  // cudaDeviceSynchronize();
  // printf("cuda err %d\n", cudaGetLastError());
  // cudaMemcpy(largEvals.get(), d_evals, sizeof(int) * size, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 8; i++)
  // {
  //   std::cout << largEvals[i] <<std::endl;
  // }


  // sumcheck_alg1(d_evals, d_t, d_T, C, n);
  sumcheck_alg1_ref(evals, t, T, C, n);

  // for (int i = 0; i < 8; i++)
  // {
  //   std::cout << T[i] <<std::endl;
  // }

  // cudaMemcpy(T, d_T, sizeof(int) * 8, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 8; i++)
  {
    std::cout << T[i] <<std::endl;
  }

  return 0;
}