/*

Copyright (c) 2023 Yrrid Software, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the �Software�), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED �AS IS�, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/cuda_utils.cuh"
#include <chrono>
#include <iostream>
#include <vector>
#include "../../curves/bls12_377/curve_config.cuh"
// #include "../../curves/bn254/curve_config.cuh"

#include <stdio.h>
#include <stdint.h>
#include <cooperative_groups.h>

using namespace BLS12_377;
// using namespace BN254;

class Dummy_Scalar
{
public:
  static constexpr unsigned NBITS = 32;

  unsigned x;
  unsigned p = 10;
  // unsigned p = 1<<30;

  static HOST_DEVICE_INLINE Dummy_Scalar zero() { return {0}; }

  static HOST_DEVICE_INLINE Dummy_Scalar one() { return {1}; }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar)
  {
    os << scalar.x;
    return os;
  }

  HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width)
  {
    return (x >> (digit_num * digit_width)) & ((1 << digit_width) - 1);
  }

  friend HOST_DEVICE_INLINE Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar& p2)
  {
    return {(p1.x + p2.x) % p1.p};
  }

    friend HOST_DEVICE_INLINE Dummy_Scalar operator-(Dummy_Scalar p1, const Dummy_Scalar& p2)
  {
    return {(p1.x - p2.x) % p1.p};
  }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const Dummy_Scalar& p2) { return (p1.x == p2.x); }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const unsigned p2) { return (p1.x == p2); }

  static HOST_DEVICE_INLINE Dummy_Scalar neg(const Dummy_Scalar& scalar) { return {scalar.p - scalar.x}; }
  static HOST_INLINE Dummy_Scalar rand_host()
  {
    return {(unsigned)rand() % 10};
    // return {(unsigned)rand()};
  }
};


// switch between dummy and real:

typedef scalar_t test_scalar;
// typedef Dummy_Scalar test_scalar;

#define $CUDA(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, cudaGetLastError()); exit(1); }

__managed__ uint32_t nextCounter[1024]={0};

#include "kernel_ntt.cu"

// uint64_t random_sample() {
//   uint64_t x;

//   x=rand() & 0xFFFF;
//   x=(x<<16) + (rand() & 0xFFFF);
//   x=(x<<16) + (rand() & 0xFFFF);
//   x=(x<<16) + (rand() & 0xFFFF);
//   if(x>0xFFFFFFFF00000001ull)
//     x=x + 0xFFFFFFFFull;
//   return x;
// }

// void random_samples(uint64_t* res, uint32_t count) {
//   for(int i=0;i<count;i++)
//     res[i]=random_sample();
// }

#define CUDA_CHECK_ERROR() \
do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void random_samples(test_scalar* res, uint32_t count) {
  for(int i=0;i<count;i++)
    res[i]=test_scalar::rand_host();
}

int main(int argc, const char** argv) {
  uint32_t    ntts=1024, repeatCount;
  test_scalar*   cpuData;
  test_scalar*   gpuData;
  cudaEvent_t start, stop;
  float       time;
  
  if(argc!=3) {
    fprintf(stderr, "Usage:  %s <nttCount> <repeatCount>\n", argv[0]);
    fprintf(stderr, "Where <nttCount> is the number of 1024-point NTTs to run in each kernel launch\n");
    fprintf(stderr, "and <repeatCount> is the number of times to run the kernel\n");
    return -1;
  }

  ntts=atoi(argv[1]);
  repeatCount=atoi(argv[2]);

  cpuData=(test_scalar*)malloc(sizeof(test_scalar)*ntts*1024);
  if(cpuData==NULL) {
    fprintf(stderr, "Malloc failed\n");
    exit(1);
  }

  std::cout << test_scalar::modulus() <<std::endl;
  std::cout<<std::endl;
  std::cout << test_scalar::omega(0) <<std::endl;
  std::cout << test_scalar::omega(1) <<std::endl;
  std::cout << test_scalar::omega(2) <<std::endl;
  std::cout << test_scalar::omega(3) <<std::endl;
  std::cout << test_scalar::omega(4) <<std::endl;
  std::cout<<std::endl;
  std::cout << test_scalar::modulus() - test_scalar::omega(0) <<std::endl;
  std::cout << test_scalar::modulus() - test_scalar::omega(1) <<std::endl;
  std::cout << test_scalar::modulus() - test_scalar::omega(2) <<std::endl;
  std::cout << test_scalar::modulus() - test_scalar::omega(3) <<std::endl;
  std::cout << test_scalar::modulus() - test_scalar::omega(4) <<std::endl;


  random_samples(cpuData, ntts*1024);

  $CUDA(cudaFuncSetAttribute(ntt1024, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024));

  for(int i=0;i<1024;i++)
    nextCounter[i]=0;
    
  fprintf(stderr, "Running with %d ntts and %d repeatCount\n", ntts, repeatCount);
  fprintf(stderr, "Warm up run\n");
  $CUDA(cudaMalloc((void**)&gpuData, sizeof(test_scalar)*ntts*1024));
  // ntt1024<<<60, 32, 97*1024>>>(gpuData, gpuData, nextCounter, ntts);
  // thread_ntt_kernel<<<1, 64, 1024>>>(gpuData, gpuData, nextCounter, ntts);
  $CUDA(cudaDeviceSynchronize());

  fprintf(stderr, "Copying data to GPU\n");
  $CUDA(cudaMemcpy(gpuData, cpuData, sizeof(test_scalar)*ntts*1024, cudaMemcpyHostToDevice));
  fprintf(stderr, "Running kernel\n");

  $CUDA(cudaEventCreate(&start));
  $CUDA(cudaEventCreate(&stop));
  $CUDA(cudaEventRecord(start, 0));

  // std::cout<<std::endl;
  // std::cout <<cpuData[0]<<std::endl;
  // std::cout <<cpuData[1]<<std::endl;
  // std::cout <<cpuData[0] + cpuData[1]<<std::endl;
  // cpuData[0] = cpuData[0] + cpuData[1];
  // std::cout <<cpuData[0]<<std::endl;
  // std::cout<<std::endl;

  printf("input\n");
  for(int i=0;i<16;i++){
    if (i%16 == 0) printf("\n");
    std::cout <<cpuData[i]<<std::endl;
  }

  // for(int i=0;i<repeatCount;i++) 
    // ntt1024<<<60, 32, 97*1024>>>(gpuData, gpuData, nextCounter, ntts);
  thread_ntt_kernel<<<1, 64, 1024>>>(gpuData, gpuData, nextCounter, ntts);
  $CUDA(cudaEventRecord(stop, 0));
  $CUDA(cudaDeviceSynchronize());
  $CUDA(cudaEventElapsedTime(&time, start, stop));
  CUDA_CHECK_ERROR();
  // printf("cuda err %d\n",cudaGetLastError());
  if(cudaGetLastError()!=0) {
    printf("cuda errrrr %d",cudaGetLastError());
    fprintf(stderr, "Error == %d\n", cudaGetLastError());
    exit(1);
  }
  fprintf(stderr, "Runtime=%0.3f MS\n", time);
  
  fprintf(stderr, "Run complete - copying data back to CPU\n");
  $CUDA(cudaMemcpy(cpuData, gpuData, sizeof(test_scalar)*ntts*1024, cudaMemcpyDeviceToHost));
  
  #if !defined(COMPUTE_ONLY)
  printf("output\n");
  for(int i=0;i<16;i++){
    if (i%16 == 0) printf("\n");
    std::cout <<cpuData[i]<<std::endl;
  }
    // for(int i=0;i<ntts*1024;i+=4) 
    //   printf("%016lX %016lX %016lX %016lX\n", cpuData[i], cpuData[i+1], cpuData[i+2], cpuData[i+3]);
  #endif
}
