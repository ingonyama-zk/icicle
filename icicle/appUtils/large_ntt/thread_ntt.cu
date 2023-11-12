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
#include "asm.cu"
#include "Sampled96.cu"

typedef Sampled96 Math;

class NTTEngine32 {
  public:
  typedef Math::Value Value;
  
  uint64_t threadRoot;
  Value    X[32];

  __device__ __forceinline__ void initializeRoot() {
    threadRoot=root1024(threadIdx.x & 0x1F);
  }
  
  __device__ __forceinline__ void loadGlobalData(uint64_t* data, uint32_t dataIndex) {
    data+=1024*dataIndex + (threadIdx.x & 0x1F);
    
    #pragma unroll
    for(uint32_t i=0;i<32;i++) 
      X[i]=Math::getNormalizedValue(data[i*32]);
  }
  
  __device__ __forceinline__ void storeGlobalData(uint64_t* data, uint32_t dataIndex) {
    data+=1024*dataIndex + (threadIdx.x & 0x1F);
      
    // samples are 4x8 transposed in registers
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      #pragma unroll
      for(uint32_t j=0;j<4;j++) 
        data[(i*4+j)*32]=make_wide(X[i+j*8].x, X[i+j*8].y);           // note transpose here
    }
  }
      
  template<uint32_t to, uint32_t from>
  __device__ __forceinline__ void copy8() {
    #pragma unroll
    for(uint32_t i=0;i<8;i++)
      X[to+i]=X[from+i];
  }
  
  __device__ __forceinline__ void ntt4(Value& X0, Value& X1, Value& X2, Value& X3) {
    Value T;

    T  = Math::add(X0, X2);
    X2 = Math::sub(X0, X2);
    X0 = Math::add(X1, X3);
    X1 = Math::sub(X1, X3);   // T has X0, X0 has X1, X2 has X2, X1 has X3
  
    X1 = Math::shiftRoot8<2>(X1);
  
    X3 = Math::sub(X2, X1);
    X1 = Math::add(X2, X1);
    X2 = Math::sub(T, X0);
    X0 = Math::add(T, X0);
  }

  __device__ __forceinline__ void ntt8(Value& X0, Value& X1, Value& X2, Value& X3, 
                                       Value& X4, Value& X5, Value& X6, Value& X7) {
    Value T;

    // out of 56,623,104 possible mappings, we have:
    T  = Math::sub(X3, X7);
    X7 = Math::add(X3, X7);
    X3 = Math::sub(X1, X5);
    X5 = Math::add(X1, X5);
    X1 = Math::add(X2, X6);
    X2 = Math::sub(X2, X6);
    X6 = Math::add(X0, X4);
    X0 = Math::sub(X0, X4);
  
    T  = Math::shiftRoot8<2>(T);
    X2 = Math::shiftRoot8<2>(X2);
    
    X4 = Math::add(X6, X1);
    X6 = Math::sub(X6, X1);
    X1 = Math::add(X3, T);
    X3 = Math::sub(X3, T);
    T  = Math::add(X5, X7);
    X5 = Math::sub(X5, X7);
    X7 = Math::add(X0, X2);
    X0 = Math::sub(X0, X2);
  
    X1 = Math::shiftRoot8<1>(X1);
    X5 = Math::shiftRoot8<2>(X5);
    X3 = Math::shiftRoot8<3>(X3);
    
    X2 = Math::add(X6, X5);
    X6 = Math::sub(X6, X5);
    X5 = Math::sub(X7, X1);
    X1 = Math::add(X7, X1);
    X7 = Math::sub(X0, X3);
    X3 = Math::add(X0, X3);
    X0 = Math::add(X4, T);
    X4 = Math::sub(X4, T);    
  }
  
  __device__ __forceinline__ void ntt32() {
     #pragma unroll
     for(uint32_t i=0;i<8;i++)
       ntt4(X[i], X[i+8], X[i+16], X[i+24]);
    
     X[9]  = Math::shiftRoot32<1>(X[9]);
     X[10] = Math::shiftRoot32<2>(X[10]);
     X[11] = Math::shiftRoot32<3>(X[11]);
     X[12] = Math::shiftRoot8<1>(X[12]);
     X[13] = Math::shiftRoot32<5>(X[13]);
     X[14] = Math::shiftRoot32<6>(X[14]);
     X[15] = Math::shiftRoot32<7>(X[15]);
      
     X[17] = Math::shiftRoot32<2>(X[17]);
     X[18] = Math::shiftRoot8<1>(X[18]);
     X[19] = Math::shiftRoot32<6>(X[19]);
     X[20] = Math::shiftRoot8<2>(X[20]);
     X[21] = Math::shiftRoot32<10>(X[21]);
     X[22] = Math::shiftRoot8<3>(X[22]);
     X[23] = Math::shiftRoot32<14>(X[23]);
      
     X[25] = Math::shiftRoot32<3>(X[25]);
     X[26] = Math::shiftRoot32<6>(X[26]);
     X[27] = Math::shiftRoot32<9>(X[27]);
     X[28] = Math::shiftRoot8<3>(X[28]);
     X[29] = Math::shiftRoot32<15>(X[29]);
     X[30] = Math::shiftRoot32<18>(X[30]);
     X[31] = Math::shiftRoot32<21>(X[31]);
     
     #pragma unroll
     for(uint32_t i=0;i<32;i+=8) 
       ntt8(X[i+0], X[i+1], X[i+2], X[i+3], X[i+4], X[i+5], X[i+6], X[i+7]);
     
     // normalizing is very expensive
     #pragma unroll
     for(uint32_t i=0;i<32;i++)
       X[i]=Math::normalize(X[i]);
  }
    
  __device__ __forceinline__ void storeSharedData(uint32_t addr) {
    const uint32_t stride=blockDim.x+1;
    
    addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts
    
    #pragma unroll
    for(int32_t i=0;i<8;i++) {
      #pragma unroll
      for(int32_t j=0;j<4;j++) 
        store_shared_u64(addr + (i*4+j)*stride*8, make_wide(X[j*8+i].x, X[j*8+i].y));    // note transpose here!
    }
  }

  __device__ __forceinline__ void loadSharedDataAndTwiddle32x32(uint32_t addr) {
    const uint32_t stride=blockDim.x+1;
    uint64_t       currentRoot;
    uint64_t       load;
    
    currentRoot=1;             // For INTT, we could save the final mult step by initializing this to 1024^-1 mod p
    
    addr=addr + (threadIdx.x & 0xFFE0)*8 + (threadIdx.x & 0x1F)*stride*8;   // use odd stride to avoid bank conflicts
    
    #pragma unroll 1
    for(int32_t i=0;i<4;i++) {
      // we can't afford 32 copies of this
      #pragma unroll
      for(uint32_t j=0;j<8;j++) {
        load=load_shared_u64(addr + (i*8 + j)*8);
        X[24+j] = Math::mul<false>(load, currentRoot);                // 8 copies of this
        currentRoot = mul(currentRoot, threadRoot);                   // 8 copies of this        
      }

      switch(i) {
        case 0: copy8<0, 24>(); break;
        case 1: copy8<8, 24>(); break;
        case 2: copy8<16, 24>(); break;
        case 3: break;
      }
    }
  }

};  