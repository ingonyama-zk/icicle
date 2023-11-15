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

// #include <stdio.h>
// #include <stdint.h>
// #include "asm.cu"
// #include "Sampled96.cu"

// typedef Sampled96 Math;

#include "thread_ntt.cu"

// EXCHANGE_OFFSET = 129*64*8

#define DATA_OFFSET 0

__launch_bounds__(384)
__global__ void ntt1024(uint64_t* out, uint64_t* in, uint32_t* next, uint32_t count) {
  NTTEngine32 engine;
  uint32_t    dataIndex=0;

  #ifdef COMPUTE_ONLY
    bool        first=true;
  #endif
  
  engine.initializeRoot();
    
  while(true) {
    if((threadIdx.x & 0x1F)==0)
      dataIndex=atomicAdd(next, 1);
    dataIndex=__shfl_sync(0xFFFFFFFF, dataIndex, 0);  //send value to all threads in warp    
    if(dataIndex<count) {
      #if defined(COMPUTE_ONLY)
        if(first)
          engine.loadGlobalData(in, dataIndex);
        first=false;
      #else
        engine.loadGlobalData(in, dataIndex);
      #endif
    }
    else {
      if(dataIndex==count + (gridDim.x*blockDim.x>>5) - 1) { //didn't understand condition
        // last one to finish, reset the counter
        atomicExch(next, 0);
      }
      return;
    }
    #pragma unroll 1
    for(uint32_t phase=0;phase<2;phase++) {
      // ntt32 produces a lot of instructions, so we put this in a loop
      engine.ntt32(); 
      if(phase==0) {
        engine.storeSharedData(DATA_OFFSET);
        __syncwarp(0xFFFFFFFF);
        engine.loadSharedDataAndTwiddle32x32(DATA_OFFSET);
      }
    }
    engine.storeGlobalData(out, dataIndex);
  }
}


__global__ void thread_ntt_kernel(test_scalar* out, test_scalar* in, uint32_t* next, uint32_t count) {
  NTTEngine engine;
  uint32_t    dataIndex=blockIdx.x*blockDim.x+threadIdx.x;

  #ifdef COMPUTE_ONLY
    bool        first=true;
  #endif
  
  // engine.initializeRoot();
  engine.loadGlobalData(in, dataIndex);
  // engine.ntt4_4();
  engine.ntt16();
  // engine.X[0] = engine.X[0] + engine.X[1];
  // out[0] = out[0] + out[1];
  // out[0] = test_scalar::zero();
  // engine.storeGlobalData(out, dataIndex);
  engine.storeGlobalData16(out, dataIndex);
    
  // while(true) {
  //   if((threadIdx.x & 0x1F)==0)
  //     dataIndex=atomicAdd(next, 1);
  //   dataIndex=__shfl_sync(0xFFFFFFFF, dataIndex, 0);      
  //   if(dataIndex<count) {
  //     #if defined(COMPUTE_ONLY)
  //       if(first)
  //         engine.loadGlobalData(in, dataIndex);
  //       first=false;
  //     #else
  //       engine.loadGlobalData(in, dataIndex);
  //     #endif
  //   }
  //   else {
  //     if(dataIndex==count + (gridDim.x*blockDim.x>>5) - 1) {
  //       // last one to finish, reset the counter
  //       atomicExch(next, 0);
  //     }
  //     return;
  //   }
  //   #pragma unroll 1
  //   for(uint32_t phase=0;phase<2;phase++) {
  //     // ntt32 produces a lot of instructions, so we put this in a loop
  //     engine.ntt32(); 
  //     if(phase==0) {
  //       engine.storeSharedData(DATA_OFFSET);
  //       __syncwarp(0xFFFFFFFF);
  //       engine.loadSharedDataAndTwiddle32x32(DATA_OFFSET);
  //     }
  //   }
  //   engine.storeGlobalData(out, dataIndex);
  // }
}

