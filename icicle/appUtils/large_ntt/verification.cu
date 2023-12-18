
#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/cuda_utils.cuh"
#include <chrono>
#include <iostream>
#include <vector>
#include "../../curves/bls12_377/curve_config.cuh"
#include "ntt.cuh"
// #include "../../curves/bn254/curve_config.cuh"

// #include <stdio.h>
// #include <stdint.h>
// #include <cooperative_groups.h>

using namespace BLS12_377;
typedef scalar_t test_scalar;
#include "kernel_ntt.cu"

#define $CUDA(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, cudaGetLastError()); exit(1); }

void random_samples(test_scalar* res, uint32_t count) {
  for(int i=0;i<count;i++)
    res[i]= i<1000? test_scalar::rand_host() : res[i-1000];
    // res[i]= i<1000? test_scalar::omega(2) : res[i-1000];
}

void incremental_values(test_scalar* res, uint32_t count) {
  for(int i=0;i<count;i++)
    res[i]=i? res[i-1]+test_scalar::one() : test_scalar::zero();
}

int main(){

  int NTT_LOG_SIZE = 6;
  int TT_LOG_SIZE = 18;
  int NTT_SIZE = 1<<NTT_LOG_SIZE;
  int TT_SIZE = 1<<TT_LOG_SIZE;

  //cpu allocation
  test_scalar* cpuIcicle;
  uint4* cpuNew;
  cpuIcicle=(test_scalar*)malloc(sizeof(test_scalar)*NTT_SIZE);
  cpuNew=(uint4*)malloc(sizeof(uint4)*NTT_SIZE*2);
  if(cpuIcicle==NULL || cpuNew==NULL) {
    fprintf(stderr, "Malloc failed\n");
    exit(1);
  }

  //gpu allocation
  test_scalar* gpuIcicle;
  uint4* gpuNew;
  uint4* gpuTwiddles;
  $CUDA(cudaMalloc((void**)&gpuIcicle, sizeof(test_scalar)*NTT_SIZE));
  $CUDA(cudaMalloc((void**)&gpuNew, sizeof(uint4)*NTT_SIZE*2));
  $CUDA(cudaMalloc((void**)&gpuTwiddles, sizeof(uint4)*TT_SIZE*2));

  //init inputs
  random_samples(cpuIcicle, NTT_SIZE);
  // incremental_values(cpuIcicle, NTT_SIZE);
  for (int i = 0; i < NTT_SIZE; i++)
  {
    cpuNew[i] = cpuIcicle[i].load_half(false);
    cpuNew[NTT_SIZE + i] = cpuIcicle[i].load_half(true);
  }
  $CUDA(cudaMemcpy(gpuIcicle, cpuIcicle, sizeof(test_scalar)*NTT_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(gpuNew, cpuNew, sizeof(uint4)*NTT_SIZE*2, cudaMemcpyHostToDevice));
  generate_external_twiddles(gpuTwiddles, TT_LOG_SIZE);


  //run ntts
  ntt64<<<1, 8, 512*sizeof(uint4)>>>(gpuNew, gpuNew, NTT_LOG_SIZE ,1);
  ntt_end2end_batch_template<test_scalar, test_scalar>(gpuIcicle, NTT_SIZE, NTT_SIZE, false, 0);
  reverse_order_batch(gpuIcicle, NTT_SIZE, NTT_LOG_SIZE, 1, 0);
  
  //verify
  $CUDA(cudaMemcpy(cpuIcicle, gpuIcicle, sizeof(test_scalar)*NTT_SIZE, cudaMemcpyDeviceToHost));
  $CUDA(cudaMemcpy(cpuNew, gpuNew, sizeof(uint4)*NTT_SIZE*2, cudaMemcpyDeviceToHost));
  bool success = true;
  for (int i = 0; i < NTT_SIZE; i++)
  {
    test_scalar icicle_temp, new_temp;
    icicle_temp = cpuIcicle[i];
    new_temp.store_half(cpuNew[i], false);
    new_temp.store_half(cpuNew[i+NTT_SIZE], true);
    if (icicle_temp != new_temp){
      success = false;
      std::cout << "ref "<< icicle_temp << " != " << new_temp <<std::endl;
    }
    // else{
    //   std::cout << "ref "<< icicle_temp << " == " << new_temp <<std::endl;
    // }
  }
  if (success){
    printf("success!\n");
  }


  return 0;

}