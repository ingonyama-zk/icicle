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

#include <stdio.h>
#include <stdint.h>
#include <cooperative_groups.h>

#define $CUDA(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, cudaGetLastError()); exit(1); }

__managed__ uint32_t nextCounter[1024]={0};

#include "kernel_ntt.cu"

uint64_t random_sample() {
  uint64_t x;

  x=rand() & 0xFFFF;
  x=(x<<16) + (rand() & 0xFFFF);
  x=(x<<16) + (rand() & 0xFFFF);
  x=(x<<16) + (rand() & 0xFFFF);
  if(x>0xFFFFFFFF00000001ull)
    x=x + 0xFFFFFFFFull;
  return x;
}

void random_samples(uint64_t* res, uint32_t count) {
  for(int i=0;i<count;i++)
    res[i]=random_sample();
}

int main(int argc, const char** argv) {
  uint32_t    ntts=1024, repeatCount;
  uint64_t*   cpuData;
  uint64_t*   gpuData;
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

  cpuData=(uint64_t*)malloc(sizeof(uint64_t)*ntts*1024);
  if(cpuData==NULL) {
    fprintf(stderr, "Malloc failed\n");
    exit(1);
  }

  random_samples(cpuData, ntts*1024);

  $CUDA(cudaFuncSetAttribute(ntt1024, cudaFuncAttributeMaxDynamicSharedMemorySize, 97*1024));

  for(int i=0;i<1024;i++)
    nextCounter[i]=0;
    
  fprintf(stderr, "Running with %d ntts and %d repeatCount\n", ntts, repeatCount);
  fprintf(stderr, "Warm up run\n");
  $CUDA(cudaMalloc((void**)&gpuData, sizeof(uint64_t)*ntts*1024));
  ntt1024<<<60, 256, 97*1024>>>(gpuData, gpuData, nextCounter, ntts);
  $CUDA(cudaDeviceSynchronize());

  fprintf(stderr, "Copying data to GPU\n");
  $CUDA(cudaMemcpy(gpuData, cpuData, sizeof(uint64_t)*ntts*1024, cudaMemcpyHostToDevice));
  fprintf(stderr, "Running kernel\n");

  $CUDA(cudaEventCreate(&start));
  $CUDA(cudaEventCreate(&stop));
  $CUDA(cudaEventRecord(start, 0));
  for(int i=0;i<repeatCount;i++) 
    ntt1024<<<60, 384, 97*1024>>>(gpuData, gpuData, nextCounter, ntts);
  $CUDA(cudaEventRecord(stop, 0));
  $CUDA(cudaDeviceSynchronize());
  $CUDA(cudaEventElapsedTime(&time, start, stop));

  if(cudaGetLastError()!=0) {
    fprintf(stderr, "Error == %d\n", cudaGetLastError());
    exit(1);
  }
  fprintf(stderr, "Runtime=%0.3f MS\n", time);
  
  fprintf(stderr, "Run complete - copying data back to CPU\n");
  $CUDA(cudaMemcpy(cpuData, gpuData, sizeof(uint64_t)*ntts*1024, cudaMemcpyDeviceToHost));
  
  #if !defined(COMPUTE_ONLY)
    for(int i=0;i<ntts*1024;i+=4) 
      printf("%016lX %016lX %016lX %016lX\n", cpuData[i], cpuData[i+1], cpuData[i+2], cpuData[i+3]);
  #endif
}
