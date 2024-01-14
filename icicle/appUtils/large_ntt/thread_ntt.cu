#ifndef T_NTT
#define T_NTT
#pragma once

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

struct stage_metadata
{
  // uint32_t stage_num;
  // uint32_t data_stride;
  uint32_t th_stride;
  uint32_t ntt_block_size;
  uint32_t ntt_block_id;
  uint32_t ntt_inp_id;
  // uint32_t ntt_meta_block_size;
  // uint32_t ntt_meta_block_id;
  // uint32_t ntt_meta_inp_id;
};

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
      X[i]=Math::getNormalizedValue(data[i*32]); //how is it fine to read strided from global?
  }
  
  __device__ __forceinline__ void storeGlobalData(uint64_t* data, uint32_t dataIndex) {
    data+=1024*dataIndex + (threadIdx.x & 0x1F);
      
    // samples are 4x8 transposed in registers
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      #pragma unroll
      for(uint32_t j=0;j<4;j++) 
        data[(i*4+j)*32]=make_wide(X[i+j*8].x, X[i+j*8].y);           // note transpose here (double traspose)
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

      switch(i) { //why?
        case 0: copy8<0, 24>(); break;
        case 1: copy8<8, 24>(); break;
        case 2: copy8<16, 24>(); break;
        case 3: break;
      }
    }
  }

};  

uint32_t constexpr STAGE_SIZES_HOST[31][5] = {{0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {4,0,0,0,0},
                              {5,0,0,0,0}, //5
                              {6,0,0,0,0}, //6
                              {0,0,0,0,0},
                              {4,4,0,0,0},
                              {0,0,0,0,0},
                              {6,4,0,0,0}, //10
                              {6,5,0,0,0}, //11
                              {6,6,0,0,0}, //12
                              {5,4,4,0,0},
                              {6,4,4,0,0},
                              {6,5,4,0,0}, //15
                              {6,5,5,0,0}, //16
                              {6,6,5,0,0}, //17
                              {6,6,6,0,0}, //18
                              {6,5,4,4,0},
                              {5,5,5,5,0}, //20
                              {6,5,5,5,0},
                              {6,6,5,5,0},
                              {6,6,6,5,0},
                              {6,6,6,6,0}, //24
                              {6,5,5,5,4},
                              {6,5,5,5,5},
                              {6,6,5,5,5},
                              {6,6,6,5,5},
                              {6,6,6,6,5},
                              {6,6,6,6,6}}; //30

__device__ constexpr uint32_t STAGE_SIZES_DEVICE[31][5] = {{0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {4,0,0,0,0},
                              {5,0,0,0,0}, //5
                              {6,0,0,0,0}, //6
                              {0,0,0,0,0},
                              {4,4,0,0,0},
                              {0,0,0,0,0},
                              {6,4,0,0,0}, //10
                              {6,5,0,0,0}, //11
                              {6,6,0,0,0}, //12
                              {5,4,4,0,0},
                              {6,4,4,0,0},
                              {6,5,4,0,0}, //15
                              {6,5,5,0,0}, //16
                              {6,6,5,0,0}, //17
                              {6,6,6,0,0}, //18
                              {6,5,4,4,0},
                              {5,5,5,5,0}, //20
                              {6,5,5,5,0},
                              {6,6,5,5,0},
                              {6,6,6,5,0},
                              {6,6,6,6,0}, //24
                              {6,5,5,5,4},
                              {6,5,5,5,5},
                              {6,6,5,5,5},
                              {6,6,6,5,5},
                              {6,6,6,6,5},
                              {6,6,6,6,6}}; //30
// __device__ uint32_t LOW_W_OFFSETS[4] = {0, 2<<12, (2<<12) + (2<<18), (2<<12) + (2<<18) + (2<<24)};
// __device__ uint32_t HIGH_W_OFFSETS[4] = {1<<12, (2<<12) + (1<<18), (2<<12) + (2<<18) + (1<<24), (2<<12) + (2<<18) + (2<<24) + (1<<30)};
__device__ constexpr uint32_t LOW_W_OFFSETS[31][5] = {{0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0}, //5
                              {0,0,0,0,0}, //6
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0}, //10
                              {0,0,0,0,0}, //11
                              {0,0,0,0,0}, //12
                              {0,0,2<<9,0,0},
                              {0,0,2<<10,0,0},
                              {0,0,2<<11,0,0}, //15
                              {0,0,2<<11,0,0}, //16
                              {0,0,2<<12,0,0}, //17
                              {0,0,2<<12,0,0}, //18
                              {0,0,2<<11,(2<<11)+(2<<15),0},
                              {0,0,2<<10,(2<<10)+(2<<15),0}, //20
                              {0,0,2<<11,(2<<11)+(2<<16),0},
                              {0,0,2<<12,(2<<12)+(2<<17),0},
                              {0,0,2<<12,(2<<12)+(2<<18),0},
                              {0,0,2<<12,(2<<12)+(2<<18),0}, //24
                              {0,0,2<<11,(2<<11)+(2<<16),(2<<11)+(2<<16)+(2<<21)},
                              {0,0,2<<11,(2<<11)+(2<<16),(2<<11)+(2<<16)+(2<<21)},
                              {0,0,2<<12,(2<<12)+(2<<17),(2<<12)+(2<<17)+(2<<22)},
                              {0,0,2<<12,(2<<12)+(2<<18),(2<<12)+(2<<18)+(2<<23)},
                              {0,0,2<<12,(2<<12)+(2<<18),(2<<12)+(2<<18)+(2<<24)},
                              {0,0,2<<12,(2<<12)+(2<<18),(2<<12)+(2<<18)+(2<<24)}}; //30
__device__ constexpr uint32_t HIGH_W_OFFSETS[31][5] = {{0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0},
                              {0,0,0,0,0}, //5
                              {0,0,0,0,0}, //6
                              {0,0,0,0,0},
                              {0,1<<8,0,0,0},
                              {0,0,0,0,0},
                              {0,1<<10,0,0,0}, //10
                              {0,1<<11,0,0,0}, //11
                              {0,1<<12,0,0,0}, //12
                              {0,1<<9,(2<<9)+(1<<13),0,0},
                              {0,1<<10,(2<<10)+(1<<14),0,0},
                              {0,1<<11,(2<<11)+(1<<15),0,0}, //15
                              {0,1<<11,(2<<11)+(1<<16),0,0}, //16
                              {0,1<<12,(2<<12)+(1<<17),0,0}, //17
                              {0,1<<12,(2<<12)+(1<<18),0,0}, //18
                              {0,1<<11,(2<<11)+(1<<15),(2<<11)+(2<<15)+(1<<19),0},
                              {0,1<<10,(2<<10)+(1<<15),(2<<10)+(2<<15)+(1<<20),0}, //20
                              {0,1<<11,(2<<11)+(1<<16),(2<<11)+(2<<16)+(1<<21),0},
                              {0,1<<12,(2<<12)+(1<<17),(2<<12)+(2<<17)+(1<<22),0},
                              {0,1<<12,(2<<12)+(1<<18),(2<<12)+(2<<18)+(1<<23),0},
                              {0,1<<12,(2<<12)+(1<<18),(2<<12)+(2<<18)+(1<<24),0}, //24
                              {0,1<<11,(2<<11)+(1<<16),(2<<11)+(2<<16)+(1<<21),(2<<11)+(2<<16)+(2<<21)+(1<<25)},
                              {0,1<<11,(2<<11)+(1<<16),(2<<11)+(2<<16)+(1<<21),(2<<11)+(2<<16)+(2<<21)+(1<<26)},
                              {0,1<<12,(2<<12)+(1<<17),(2<<12)+(2<<17)+(1<<22),(2<<12)+(2<<17)+(2<<22)+(1<<27)},
                              {0,1<<12,(2<<12)+(1<<18),(2<<12)+(2<<18)+(1<<23),(2<<12)+(2<<18)+(2<<23)+(1<<28)},
                              {0,1<<12,(2<<12)+(1<<18),(2<<12)+(2<<18)+(1<<24),(2<<12)+(2<<18)+(2<<24)+(1<<29)},
                              {0,1<<12,(2<<12)+(1<<18),(2<<12)+(2<<18)+(1<<24),(2<<12)+(2<<18)+(2<<24)+(1<<30)}}; //30

class NTTEngine {
  public:
  // typedef Math::Value Value;
  
  // uint64_t threadRoot;
  test_scalar    X[8];
  test_scalar    WB[3];
  test_scalar    WI[7];
  test_scalar    WE[8];

  // __device__ __forceinline__ void initializeRoot() {
  //   threadRoot=root1024(threadIdx.x & 0x1F);
  // }
  __device__ __forceinline__ void initializeRoot(bool stride) {
    #pragma unroll
    for (int i = 0; i < 7; i++)
    {
      WI[i] = test_scalar::omega8(((stride?(threadIdx.x>>3):(threadIdx.x))&0x7)*(i+1));
      // if (blockIdx.x == 0 && threadIdx.x == 8) printf("%d\n",WI[i].load_half(false).w);
    }
  }

// __device__ __forceinline__ void loadBasicTwiddles(bool inv){
    
//     // #pragma unroll
//     // for (int i = 1; i < 4; i++)
//     // {
//       // WB[0] = inv? test_scalar::omega_inv(3) : test_scalar::omega(3);
//       WB[0] = inv? test_scalar::omega4_inv(4) : test_scalar::omega4(4);
//       WB[1] = inv? test_scalar::win3_inv(6) : test_scalar::win3(6);
//       WB[2] = inv? test_scalar::win3_inv(7) : test_scalar::win3(7);
//     // }
//   }

  __device__ __forceinline__ void loadBasicTwiddles(uint4* basic_twiddles){
    
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
      WB[i].store_half(basic_twiddles[i], false);
      WB[i].store_half(basic_twiddles[i+3], true);
      // WB[0] = inv? test_scalar::omega4_inv(4) : test_scalar::omega4(4);
      // WB[1] = inv? test_scalar::win3_inv(6) : test_scalar::win3(6);
      // WB[2] = inv? test_scalar::win3_inv(7) : test_scalar::win3(7);
    }
  }

  __device__ __forceinline__ void loadInternalTwiddles(uint4* data, bool stride){
    // data += (1<<(tw_log_size-6));
    #pragma unroll
    for (int i = 0; i < 7; i++)
    {
      WI[i].store_half(data[((stride?(threadIdx.x>>3):(threadIdx.x))&0x7)*(i+1)], false);
      WI[i].store_half(data[((stride?(threadIdx.x>>3):(threadIdx.x))&0x7)*(i+1) + 64], true);
      // if (blockIdx.x == 0 && threadIdx.x == 8) printf("%d\n",WI[i].load_half(false).w);
    }
  }

  __device__ __forceinline__ void loadInternalTwiddles32(uint4* data, bool stride){
    // data += (1<<(tw_log_size-6));
    #pragma unroll
    for (int i = 0; i < 7; i++)
    {
      WI[i].store_half(data[2*((stride?(threadIdx.x>>4):(threadIdx.x))&0x3)*(i+1)], false);
      WI[i].store_half(data[2*((stride?(threadIdx.x>>4):(threadIdx.x))&0x3)*(i+1) + 64], true);
      // if (blockIdx.x == 0 && threadIdx.x == 8) printf("%d\n",WI[i].load_half(false).w);
    }
  }

  __device__ __forceinline__ void loadInternalTwiddles16(uint4* data, bool stride){
    // data += (1<<(tw_log_size-6));
    #pragma unroll
    for (int i = 0; i < 7; i++)
    {
      WI[i].store_half(data[4*((stride?(threadIdx.x>>5):(threadIdx.x))&0x1)*(i+1)], false);
      WI[i].store_half(data[4*((stride?(threadIdx.x>>5):(threadIdx.x))&0x1)*(i+1) + 64], true);
      // if (blockIdx.x == 0 && threadIdx.x == 8) printf("%d\n",WI[i].load_half(false).w);
    }
  }

  __device__ __forceinline__ void loadGlobalDataDep(test_scalar* data, uint32_t dataIndex) {
    // data+=1024*dataIndex + (threadIdx.x & 0x1F);
    data += 16*dataIndex;
      
    // samples are 4x8 transposed in registers
    #pragma unroll
    for(uint32_t i=0;i<16;i++) {
      X[i] = data[i];           // note transpose here
    }
  }

   __device__ __forceinline__ void storeGlobalDataDep(test_scalar* data, uint32_t dataIndex) {
    // data+=1024*dataIndex + (threadIdx.x & 0x1F);
    data += 16*dataIndex;
      
    // samples are 4x8 transposed in registers
    #pragma unroll
    for(uint32_t i=0;i<16;i++) {
      data[i]=X[i];           // note transpose here
    }
  }
  
  // __device__ __forceinline__ void loadInternalTwiddles(uint4* data, uint32_t block_offset, uint32_t stride, uint32_t high_bits_offset) {
    
  //   if (stride == 1){
  //     data += block_offset + (threadIdx.x & 0x7) + 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
  //   }
  //   else {
  //     data += block_offset + (threadIdx.x & 0x7) + stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
  //   }

  //   #pragma unroll
  //   for(uint32_t i=0;i<7;i++) {
  //     WI[i].store_half(data[8*i*stride], false);
  //     WI[i].store_half(data[8*i*stride + high_bits_offset], true);
  //   }
  // }

  // __device__ __forceinline__ void loadExternalTwiddles(uint4* data, uint32_t block_offset, uint32_t stride, uint32_t high_bits_offset) {
  __device__ __forceinline__ void loadExternalTwiddles(uint4* data, uint32_t tw_stride, bool strided, stage_metadata s_meta, uint32_t log_size, uint32_t stage_num) {
    
    // uint32_t extra_tw = tw_log_size - log_size;
    // tw_stride = (tw_stride << 6);
    // int max_stride = 64*64*64;

    // data += max_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id%tw_stride)*(max_stride/tw_stride);
    // data += tw_stride*s_meta.ntt_inp_id + s_meta.ntt_block_id%tw_stride;
    data += tw_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id&(tw_stride-1));
    // if (tw_stride == max_stride){
      // data += (s_meta.ntt_meta_inp_id*blockDim.x) % (s_meta.ntt_meta_block_size*blockDim.x*blockDim.x);
      // tw_stride = (tw_stride << extra_tw);
      // data += 8 * (blockIdx.x & 0x7) + (threadIdx.x & 0x7) + tw_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    // }
    // else {
    //   data += s_meta.ntt_meta_inp_id*(max_stride/tw_stride) + (s_meta.ntt_block_id%64)*max_stride;
      // data += (s_meta.ntt_block_id*blockDim.x + s_meta.ntt_inp_id) % (s_meta.ntt_meta_block_size*blockDim.x);
      // data += (1<<extra_tw) * 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + (1<<extra_tw) * 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    // }

    // data += blockIdx.x*64*64*8 + (threadIdx.x & 0x7) + 64*64 * (threadIdx.x >> 3);
    // stride = 1;

    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      // WE[i].store_half(data[8*i*max_stride], false);
      // WE[i].store_half(data[8*i*max_stride + max_stride*64], true);
      WE[i].store_half(data[8*i*tw_stride + LOW_W_OFFSETS[log_size][stage_num]], false);
      WE[i].store_half(data[8*i*tw_stride + HIGH_W_OFFSETS[log_size][stage_num]], true);
    }
    // #pragma unroll
    // for(uint32_t i=0;i<7;i++) {
    //   WI[i].store_half(data[8*i*stride], false);
    //   WI[i].store_half(data[8*i*stride + high_bits_offset], true);
    // }
  }

  __device__ __forceinline__ void loadExternalTwiddles32(uint4* data, uint32_t tw_stride, bool strided, stage_metadata s_meta, uint32_t log_size, uint32_t stage_num) {
    
    // uint32_t extra_tw = tw_log_size - log_size;
    // tw_stride = (tw_stride << 6);
    // int max_stride = 64*64*64;

    // data += max_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id%tw_stride)*(max_stride/tw_stride);
    // data += tw_stride*s_meta.ntt_inp_id*2 + s_meta.ntt_block_id%tw_stride;
    data += tw_stride*s_meta.ntt_inp_id*2 + (s_meta.ntt_block_id&(tw_stride-1));
    // if (tw_stride ==(1<<20) && blockIdx.x == 0 && threadIdx.x == 1) printf("tw addr %d\n", tw_stride*s_meta.ntt_inp_id*2 + s_meta.ntt_block_id%tw_stride);
    // if (tw_stride == max_stride){
      // data += (s_meta.ntt_meta_inp_id*blockDim.x) % (s_meta.ntt_meta_block_size*blockDim.x*blockDim.x);
      // tw_stride = (tw_stride << extra_tw);
      // data += 8 * (blockIdx.x & 0x7) + (threadIdx.x & 0x7) + tw_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    // }
    // else {
    //   data += s_meta.ntt_meta_inp_id*(max_stride/tw_stride) + (s_meta.ntt_block_id%64)*max_stride;
      // data += (s_meta.ntt_block_id*blockDim.x + s_meta.ntt_inp_id) % (s_meta.ntt_meta_block_size*blockDim.x);
      // data += (1<<extra_tw) * 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + (1<<extra_tw) * 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    // }

    // data += blockIdx.x*64*64*8 + (threadIdx.x & 0x7) + 64*64 * (threadIdx.x >> 3);
    // stride = 1;
    #pragma unroll
    for(uint32_t j=0;j<2;j++) {
      #pragma unroll
      for(uint32_t i=0;i<4;i++) {
            // if (tw_stride ==(1<<20) && blockIdx.x == 0 && threadIdx.x == 1) printf("tw addr delta %d\n", (8*i+j)*tw_stride);
        // data[(8*i+j)*data_stride] = X[4*j+i].load_half(false);
        // data[(8*i+j)*data_stride + (1<<log_size)] = X[4*j+i].load_half(true);
        WE[4*j+i].store_half(data[(8*i+j)*tw_stride + LOW_W_OFFSETS[log_size][stage_num]], false);
        WE[4*j+i].store_half(data[(8*i+j)*tw_stride + HIGH_W_OFFSETS[log_size][stage_num]], true);
        // if (tw_stride ==(1<<20) && blockIdx.x == 0 && threadIdx.x == 1) printf("tw  %d\n", WE[4*j+i]);
      }
    }
    
    // #pragma unroll
    // for(uint32_t i=0;i<8;i++) {
    //   // WE[i].store_half(data[8*i*max_stride], false);
    //   // WE[i].store_half(data[8*i*max_stride + max_stride*64], true);
    //   WE[i].store_half(data[8*i*tw_stride + LOW_W_OFFSETS[log_size][stage_num]], false);
    //   WE[i].store_half(data[8*i*tw_stride + HIGH_W_OFFSETS[log_size][stage_num]], true);
    // }
    // #pragma unroll
    // for(uint32_t i=0;i<7;i++) {
    //   WI[i].store_half(data[8*i*stride], false);
    //   WI[i].store_half(data[8*i*stride + high_bits_offset], true);
    // }
  }

  __device__ __forceinline__ void loadExternalTwiddles16(uint4* data, uint32_t tw_stride, bool strided, stage_metadata s_meta, uint32_t log_size, uint32_t stage_num) {
    
    // uint32_t extra_tw = tw_log_size - log_size;
    // tw_stride = (tw_stride << 6);
    // int max_stride = 64*64*64;

    // data += max_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id%tw_stride)*(max_stride/tw_stride);
    // data += tw_stride*s_meta.ntt_inp_id*4 + s_meta.ntt_block_id%tw_stride;
    data += tw_stride*s_meta.ntt_inp_id*4 + (s_meta.ntt_block_id&(tw_stride-1));
    // if (tw_stride ==(1<<20) && blockIdx.x == 0 && threadIdx.x == 1) printf("tw addr %d\n", tw_stride*s_meta.ntt_inp_id*2 + s_meta.ntt_block_id%tw_stride);
    // if (tw_stride == max_stride){
      // data += (s_meta.ntt_meta_inp_id*blockDim.x) % (s_meta.ntt_meta_block_size*blockDim.x*blockDim.x);
      // tw_stride = (tw_stride << extra_tw);
      // data += 8 * (blockIdx.x & 0x7) + (threadIdx.x & 0x7) + tw_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    // }
    // else {
    //   data += s_meta.ntt_meta_inp_id*(max_stride/tw_stride) + (s_meta.ntt_block_id%64)*max_stride;
      // data += (s_meta.ntt_block_id*blockDim.x + s_meta.ntt_inp_id) % (s_meta.ntt_meta_block_size*blockDim.x);
      // data += (1<<extra_tw) * 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + (1<<extra_tw) * 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    // }

    // data += blockIdx.x*64*64*8 + (threadIdx.x & 0x7) + 64*64 * (threadIdx.x >> 3);
    // stride = 1;
    #pragma unroll
    for(uint32_t j=0;j<4;j++) {
      #pragma unroll
      for(uint32_t i=0;i<2;i++) {
            // if (tw_stride ==(1<<20) && blockIdx.x == 0 && threadIdx.x == 1) printf("tw addr delta %d\n", (8*i+j)*tw_stride);
        // data[(8*i+j)*data_stride] = X[4*j+i].load_half(false);
        // data[(8*i+j)*data_stride + (1<<log_size)] = X[4*j+i].load_half(true);
        WE[2*j+i].store_half(data[(8*i+j)*tw_stride + LOW_W_OFFSETS[log_size][stage_num]], false);
        WE[2*j+i].store_half(data[(8*i+j)*tw_stride + HIGH_W_OFFSETS[log_size][stage_num]], true);
        // if (tw_stride ==(1<<20) && blockIdx.x == 0 && threadIdx.x == 1) printf("tw  %d\n", WE[4*j+i]);
      }
    }
  }

  // __device__ __forceinline__ void loadGlobalData(uint4* data, uint32_t block_offset, uint32_t stride, uint32_t high_bits_offset) {
  __device__ __forceinline__ void loadGlobalData(uint4* data, uint32_t data_stride, uint32_t log_data_stride, uint32_t log_size, bool strided, stage_metadata s_meta) {
    
    if (strided){
      // data += s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size;
      data += (s_meta.ntt_block_id&(data_stride-1)) + data_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id>>log_data_stride)*data_stride*s_meta.ntt_block_size;
      // data += 8 * (blockIdx.x & 0x7) + ((1<<log_size)/data_stride) * (blockIdx.x >> 3) + (threadIdx.x & 0x7) + data_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    else {
      data += s_meta.ntt_block_id*s_meta.ntt_block_size + s_meta.ntt_inp_id;
      // data += 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }

    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      X[i].store_half(data[s_meta.th_stride*i*data_stride], false);
      X[i].store_half(data[s_meta.th_stride*i*data_stride + (1<<log_size)], true);
    }
    // if (threadIdx.x == 0){
    // for(uint32_t i=0;i<8;i++) {
    //   printf("%d", data[s_meta.th_stride*i*data_stride].w);
    //   printf("%d", data[s_meta.th_stride*i*data_stride+ (1<<log_size)].w);
    //   printf("\n");
    // }
    // }
    // #pragma unroll
    // for(uint32_t i=0;i<8;i++) {
    //   WE[i].store_half(data[8*i*stride], false);
    //   WE[i].store_half(data[8*i*stride + high_bits_offset], true);
    // }
    // #pragma unroll
    // for(uint32_t i=0;i<7;i++) {
    //   WI[i].store_half(data[8*i*stride], false);
    //   WI[i].store_half(data[8*i*stride + high_bits_offset], true);
    // }
    // #pragma unroll
    // for(uint32_t i=0;i<16;i++) 
    //   X[i].store_half(data[16*i*stride + high_bits_offset], true);

  }

  
  
  __device__ __forceinline__ void storeGlobalData(uint4* data, uint32_t data_stride, uint32_t log_data_stride, uint32_t log_size, bool strided, stage_metadata s_meta) {
    
    if (strided){
      // data += s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size;
      data += (s_meta.ntt_block_id&(data_stride-1)) + data_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id>>log_data_stride)*data_stride*s_meta.ntt_block_size;
      // data += 8 * (blockIdx.x & 0x7) + ((1<<log_size)/data_stride) * (blockIdx.x >> 3) + (threadIdx.x & 0x7) + data_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    else {
      data += s_meta.ntt_block_id*s_meta.ntt_block_size + s_meta.ntt_inp_id;
      // data += 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    // if (blockIdx.x<2){
    //   printf("block %d thread %d read address %d\n",blockIdx.x, threadIdx.x, block_offset + (threadIdx.x & 0xf) + 256 * (threadIdx.x >> 4));
    // }
        // if (data_stride == (1<<20) && blockIdx.x == 65537 && threadIdx.x ==1) printf("block %d store addr %d\n", blockIdx.x, s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size);

    
    // uint4 zero{0,0,0,0};
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      // if (data_stride == (32*32*32*32)){
      //   if (blockIdx.x == 0 && threadIdx.x%16==1){
      //     // continue;
      //     data[s_meta.th_stride*i*data_stride] = uint4{2,0,0,0};
      //     data[s_meta.th_stride*i*data_stride + (1<<log_size)] = uint4{3,0,0,0};
      //     continue;
      //   }
      //   data[s_meta.th_stride*i*data_stride] = WE[i].load_half(false);
      //   data[s_meta.th_stride*i*data_stride + (1<<log_size)] = WE[i].load_half(true);
      // }
      // else{
        data[s_meta.th_stride*i*data_stride] = X[i].load_half(false);
        data[s_meta.th_stride*i*data_stride + (1<<log_size)] = X[i].load_half(true);
      // }
    }
      // data[16*i*stride] = zero;
    // #pragma unroll
    // for(uint32_t i=0;i<16;i++) 
    //   data[16*i*stride + high_bits_offset] = X[i].load_half(true);
      // data[16*i*stride + high_bits_offset] = zero;

  }

  __device__ __forceinline__ void loadGlobalData32(uint4* data, uint32_t data_stride, uint32_t log_data_stride, uint32_t log_size, bool strided, stage_metadata s_meta) {
    
    if (strided){
      // data += s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id*2 + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size;
      data += (s_meta.ntt_block_id&(data_stride-1)) + data_stride*s_meta.ntt_inp_id*2 + (s_meta.ntt_block_id>>log_data_stride)*data_stride*s_meta.ntt_block_size;
      // data += 8 * (blockIdx.x & 0x7) + ((1<<log_size)/data_stride) * (blockIdx.x >> 3) + (threadIdx.x & 0x7) + data_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    else {
      data += s_meta.ntt_block_id*s_meta.ntt_block_size + s_meta.ntt_inp_id*2;
      // data += 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    // if (data_stride == (1<<20) && blockIdx.x == 65537 && threadIdx.x ==1) printf("block %d load addr %d\n", blockIdx.x, s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id*2 + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size);

    #pragma unroll
    for(uint32_t j=0;j<2;j++) {
      #pragma unroll
      for(uint32_t i=0;i<4;i++) {
        // if (data_stride == (1<<20) && blockIdx.x == 65537 && threadIdx.x ==1) printf("%d tttt\n", (8*i+j)*data_stride);
        X[4*j+i].store_half(data[(8*i+j)*data_stride], false);
        X[4*j+i].store_half(data[(8*i+j)*data_stride + (1<<log_size)], true);
      }
    }

    
    // #pragma unroll
    // for(uint32_t i=0;i<8;i++) {
    //   X[i].store_half(data[s_meta.th_stride*i*data_stride], false);
    //   X[i].store_half(data[s_meta.th_stride*i*data_stride + (1<<log_size)], true);
    // }
  }

   __device__ __forceinline__ void storeGlobalData32(uint4* data, uint32_t data_stride, uint32_t log_data_stride, uint32_t log_size, bool strided, stage_metadata s_meta) {
    
    if (strided){
      // data += s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id*2 + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size;
      data += (s_meta.ntt_block_id&(data_stride-1)) + data_stride*s_meta.ntt_inp_id*2 + (s_meta.ntt_block_id>>log_data_stride)*data_stride*s_meta.ntt_block_size;
      // data += 8 * (blockIdx.x & 0x7) + ((1<<log_size)/data_stride) * (blockIdx.x >> 3) + (threadIdx.x & 0x7) + data_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    else {
      data += s_meta.ntt_block_id*s_meta.ntt_block_size + s_meta.ntt_inp_id*2;
      // data += 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    // if (blockIdx.x<2){
    //   printf("block %d thread %d read address %d\n",blockIdx.x, threadIdx.x, block_offset + (threadIdx.x & 0xf) + 256 * (threadIdx.x >> 4));
    // }
    
    // uint4 zero{0,0,0,0};
    #pragma unroll
    for(uint32_t j=0;j<2;j++) {
      #pragma unroll
      for(uint32_t i=0;i<4;i++) {
        // if (data_stride == 32*32*32*32){
        //   // if (threadIdx.x%16==1){
        //     data[(8*i+j)*data_stride] = uint4{2,0,0,0};
        //     data[(8*i+j)*data_stride + (1<<log_size)] = uint4{3,0,0,0};
        //     continue;
        //   // }
        //   data[(8*i+j)*data_stride] = WE[4*j+i].load_half(false);
        //   data[(8*i+j)*data_stride + (1<<log_size)] = WE[4*j+i].load_half(true);
        // }
        // else{
          data[(8*i+j)*data_stride] = X[4*j+i].load_half(false);
          data[(8*i+j)*data_stride + (1<<log_size)] = X[4*j+i].load_half(true);
        // }
      }
    }
    // if (threadIdx.x == 0){
    //   #pragma unroll
    // for(uint32_t j=0;j<2;j++) {
    //   #pragma unroll
    //   for(uint32_t i=0;i<4;i++) {
    //     // if (data_stride == 64){
    //     //   // if (i==0){
    //     //   //   data[8*i*data_stride] = uint4{2,0,0,0};
    //     //   //   data[8*i*data_stride + (1<<log_size)] = uint4{0,0,0,0};
    //     //   //   continue;
    //     //   // }
    //     //   data[8*i*data_stride] = WE[i].load_half(false);
    //     //   data[8*i*data_stride + (1<<log_size)] = WE[i].load_half(true);
    //     // }
    //     // else{
    //       printf("%d",X[4*j+i].load_half(false).w);
    //       printf("%d",X[4*j+i].load_half(true).w);
    //       printf("\n");
    //       // data[(8*i+j)*data_stride + (1<<log_size)] = X[4*j+i].load_half(true);
    //     // }
    //   }
    // }
    // }
      // data[16*i*stride] = zero;
    // #pragma unroll
    // for(uint32_t i=0;i<16;i++) 
    //   data[16*i*stride + high_bits_offset] = X[i].load_half(true);
      // data[16*i*stride + high_bits_offset] = zero;

  }


  
  __device__ __forceinline__ void loadGlobalData16(uint4* data, uint32_t data_stride, uint32_t log_data_stride, uint32_t log_size, bool strided, stage_metadata s_meta) {
    
    if (strided){
      // data += s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id*4 + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size;
      data += (s_meta.ntt_block_id&(data_stride-1)) + data_stride*s_meta.ntt_inp_id*4 + (s_meta.ntt_block_id>>log_data_stride)*data_stride*s_meta.ntt_block_size;
      // data += 8 * (blockIdx.x & 0x7) + ((1<<log_size)/data_stride) * (blockIdx.x >> 3) + (threadIdx.x & 0x7) + data_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    else {
      data += s_meta.ntt_block_id*s_meta.ntt_block_size + s_meta.ntt_inp_id*4;
      // data += 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    // if (data_stride == (1<<20) && blockIdx.x == 65537 && threadIdx.x ==1) printf("block %d load addr %d\n", blockIdx.x, s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id*2 + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size);

    #pragma unroll
    for(uint32_t j=0;j<4;j++) {
      #pragma unroll
      for(uint32_t i=0;i<2;i++) {
        // if (data_stride == (1<<20) && blockIdx.x == 65537 && threadIdx.x ==1) printf("%d tttt\n", (8*i+j)*data_stride);
        X[2*j+i].store_half(data[(8*i+j)*data_stride], false);
        X[2*j+i].store_half(data[(8*i+j)*data_stride + (1<<log_size)], true);
      }
    }

    
    // #pragma unroll
    // for(uint32_t i=0;i<8;i++) {
    //   X[i].store_half(data[s_meta.th_stride*i*data_stride], false);
    //   X[i].store_half(data[s_meta.th_stride*i*data_stride + (1<<log_size)], true);
    // }
  }

   __device__ __forceinline__ void storeGlobalData16(uint4* data, uint32_t data_stride, uint32_t log_data_stride, uint32_t log_size, bool strided, stage_metadata s_meta) {
    
    if (strided){
      // data += s_meta.ntt_block_id%data_stride + data_stride*s_meta.ntt_inp_id*4 + (s_meta.ntt_block_id/data_stride)*data_stride*s_meta.ntt_block_size;
      data += (s_meta.ntt_block_id&(data_stride-1)) + data_stride*s_meta.ntt_inp_id*4 + (s_meta.ntt_block_id>>log_data_stride)*data_stride*s_meta.ntt_block_size;
      // data += 8 * (blockIdx.x & 0x7) + ((1<<log_size)/data_stride) * (blockIdx.x >> 3) + (threadIdx.x & 0x7) + data_stride * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    else {
      data += s_meta.ntt_block_id*s_meta.ntt_block_size + s_meta.ntt_inp_id*4;
      // data += 8 * 64 * blockIdx.x + (threadIdx.x & 0x7) + 64 * (threadIdx.x >> 3); //block offset, thread offset, ntt_offset
    }
    // if (blockIdx.x<2){
    //   printf("block %d thread %d read address %d\n",blockIdx.x, threadIdx.x, block_offset + (threadIdx.x & 0xf) + 256 * (threadIdx.x >> 4));
    // }
    
    // uint4 zero{0,0,0,0};
    #pragma unroll
    for(uint32_t j=0;j<4;j++) {
      #pragma unroll
      for(uint32_t i=0;i<2;i++) {
        // if (data_stride == 32*32*32*32){
        //   // if (threadIdx.x%16==1){
        //     data[(8*i+j)*data_stride] = uint4{2,0,0,0};
        //     data[(8*i+j)*data_stride + (1<<log_size)] = uint4{3,0,0,0};
        //     continue;
        //   // }
        //   data[(8*i+j)*data_stride] = WE[4*j+i].load_half(false);
        //   data[(8*i+j)*data_stride + (1<<log_size)] = WE[4*j+i].load_half(true);
        // }
        // else{
          data[(8*i+j)*data_stride] = X[2*j+i].load_half(false);
          data[(8*i+j)*data_stride + (1<<log_size)] = X[2*j+i].load_half(true);
        // }
      }
    }
    // if (threadIdx.x == 0){
    //   #pragma unroll
    // for(uint32_t j=0;j<2;j++) {
    //   #pragma unroll
    //   for(uint32_t i=0;i<4;i++) {
    //     // if (data_stride == 64){
    //     //   // if (i==0){
    //     //   //   data[8*i*data_stride] = uint4{2,0,0,0};
    //     //   //   data[8*i*data_stride + (1<<log_size)] = uint4{0,0,0,0};
    //     //   //   continue;
    //     //   // }
    //     //   data[8*i*data_stride] = WE[i].load_half(false);
    //     //   data[8*i*data_stride + (1<<log_size)] = WE[i].load_half(true);
    //     // }
    //     // else{
    //       printf("%d",X[4*j+i].load_half(false).w);
    //       printf("%d",X[4*j+i].load_half(true).w);
    //       printf("\n");
    //       // data[(8*i+j)*data_stride + (1<<log_size)] = X[4*j+i].load_half(true);
    //     // }
    //   }
    // }
    // }
      // data[16*i*stride] = zero;
    // #pragma unroll
    // for(uint32_t i=0;i<16;i++) 
    //   data[16*i*stride + high_bits_offset] = X[i].load_half(true);
      // data[16*i*stride + high_bits_offset] = zero;

  }





  // __device__ __forceinline__ void storeGlobalData16(test_scalar* data, uint32_t dataIndex) {
  //   // data+=1024*dataIndex + (threadIdx.x & 0x1F);
  //   data += 16*dataIndex;
      
  //   // samples are 4x8 transposed in registers
  //   #pragma unroll
  //   for(uint32_t i=0;i<16;i++) {
  //     #pragma unroll
  //     for(uint32_t j=0;j<4;j++) 
  //       data[(i*4+j)]= X[i+j*4];           // note transpose here
  //   }
  // }

  // __device__ __forceinline__ void storeGlobalData8_2(test_scalar* data, uint32_t dataIndex) {
  //   // data+=1024*dataIndex + (threadIdx.x & 0x1F);
  //   data += 16*dataIndex;
      
  //   // samples are 4x8 transposed in registers
  //   #pragma unroll
  //   for(uint32_t i=0;i<2;i++) {
  //     #pragma unroll
  //     for(uint32_t j=0;j<8;j++) 
  //       data[(i*8+j)]= X[i+j*2];           // note transpose here
  //   }
  // }
      
  // template<uint32_t to, uint32_t from>
  // __device__ __forceinline__ void copy8() {
  //   #pragma unroll
  //   for(uint32_t i=0;i<8;i++)
  //     X[to+i]=X[from+i];
  // }

  __device__ __forceinline__ void ntt4_2() {
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        ntt4(X[4*i], X[4*i+1], X[4*i+2], X[4*i+3]);
    }
    
  }

  __device__ __forceinline__ void ntt2_4() {
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        ntt2(X[2*i], X[2*i+1]);
    }
    
  }

  // __device__ __forceinline__ void ntt8_2() {
  //   for (int i = 0; i < 2; i++)
  //   {
  //       // ntt8(X[8*i], X[8*i+1], X[8*i+2], X[8*i+3], X[8*i+4], X[8*i+5], X[8*i+6], X[8*i+7]);
  //       ntt8win(X[8*i], X[8*i+1], X[8*i+2], X[8*i+3], X[8*i+4], X[8*i+5], X[8*i+6], X[8*i+7]);
  //   }
    
  // }

  __device__ __forceinline__ void ntt2(test_scalar& X0, test_scalar& X1) {
    test_scalar T;

    T = X0 + X1;
    X1 = X0 - X1;
    X0 = T;
  }

  __device__ __forceinline__ void ntt4(test_scalar& X0, test_scalar& X1, test_scalar& X2, test_scalar& X3) {
    test_scalar T;

    T  = X0 + X2;
    X2 = X0 - X2;
    X0 = X1 + X3;
    X1 = X1 - X3;   // T has X0, X0 has X1, X2 has X2, X1 has X3
  
    // X1 = X1 * (test_scalar::modulus() - test_scalar::omega(2));
    // X1 = X1 * test_scalar::omega4_inv(4);
    X1 = X1 * WB[0];
    // X1 = X1 * test_scalar::omega(7);
  
    X3 = X2 - X1;
    X1 = X2 + X1;
    X2 = T - X0;
    X0 = T + X0;
  }

  //rbo vertion
  __device__ __forceinline__ void ntt4rbo(test_scalar& X0, test_scalar& X1, test_scalar& X2, test_scalar& X3) {
    test_scalar T;

    T = X0 - X1;
    X0 = X0 + X1;
    X1 = X2 + X3;
    X3 = X2 - X3;   // T has X0, X0 has X1, X2 has X2, X1 has X3

    // if (blockIdx.x==0 && threadIdx.x == 0){
    //   printf(T);
    // }
  
    // X3 = X3 * (test_scalar::modulus() - test_scalar::omega(2));
    // X3 = X3 * test_scalar::omega4(4);
    X3 = X3 * WB[0];
    
    X2 = X0 - X1;
    X0 = X0 + X1;
    X1 = T + X3;
    X3 = T - X3;
  }

  __device__ __forceinline__ void ntt8(test_scalar& X0, test_scalar& X1, test_scalar& X2, test_scalar& X3, 
                                       test_scalar& X4, test_scalar& X5, test_scalar& X6, test_scalar& X7) {
    test_scalar T;

    // out of 56,623,104 possible mappings, we have:
    T  = X3 - X7;
    X7 = X3 + X7;
    X3 = X1 - X5;
    X5 = X1 + X5;
    X1 = X2 + X6;
    X2 = X2 - X6;
    X6 = X0 + X4;
    X0 = X0 - X4;
  
    // T  = T * test_scalar::omega4(4);
    // X2 = X2 * test_scalar::omega4(4);
    T  = T * WB[1];
    X2 = X2 * WB[1];
    
    X4 = X6 + X1;
    X6 = X6 - X1;
    X1 = X3 + T;
    X3 = X3 - T;
    T  = X5 + X7;
    X5 = X5 - X7;
    X7 = X0 + X2;
    X0 = X0 - X2;
  
    // X1 = X1 * test_scalar::omega4(2);
    // X5 = X5 * test_scalar::omega4(4);
    // X3 = X3 * test_scalar::omega4(6);
    X1 = X1 * WB[0];
    X5 = X5 * WB[1];
    X3 = X3 * WB[2];
    
    X2 = X6 + X5;
    X6 = X6 - X5;
    X5 = X7 - X1;
    X1 = X7 + X1;
    X7 = X0 - X3;
    X3 = X0 + X3;
    X0 = X4 + T;
    X4 = X4 - T;   
  }

  __device__ __forceinline__ void ntt8win() {
    test_scalar T;

    // out of 56,623,104 possible mappings, we have:
    // X7 = X7 * test_scalar::omega4(3);
    // X3 = X3 * test_scalar::omega4(3);
    T  = X[3] - X[7];
    X[7] = X[3] + X[7];
    // X1 = X1 * test_scalar::omega4(3);
    // X5 = X5 * test_scalar::omega4(3);
    X[3] = X[1] - X[5];
    X[5] = X[1] + X[5];
    // X2 = X2 * test_scalar::omega4(3);
    // X6 = X6 * test_scalar::omega4(3);
    X[1] = X[2] + X[6];
    X[2] = X[2] - X[6];
    // X0 = X0 * test_scalar::omega4(3);
    // X4 = X4 * test_scalar::omega4(3);
    X[6] = X[0] + X[4];
    X[0] = X[0] - X[4];
  
    //T  = T * test_scalar::omega4(4);
    // X[2] = X[2] * test_scalar::omega4(4);
    X[2] = X[2] * WB[0];
    
    X[4] = X[6] + X[1];
    X[6] = X[6] - X[1];
    X[1] = X[3] + T;
    X[3] = X[3] - T;
    T  = X[5] + X[7];
    X[5] = X[5] - X[7];
    X[7] = X[0] + X[2];
    X[0] = X[0] - X[2];
  
    //X1 = X1 * test_scalar::omega4(2);
    // X[1] = X[1] * test_scalar::win3(6);
    X[1] = X[1] * WB[1];
    // X[5] = X[5] * test_scalar::omega4(4) ;
    X[5] = X[5] * WB[0] ;
    //X3 = X3 * test_scalar::omega4(6);
    X[3] = X[3] * WB[2];
    // X[3] = X[3] * test_scalar::win3(7);
    
    
    X[2] = X[6] + X[5];
    X[6] = X[6] - X[5];

    X[5] = X[1] + X[3];
    X[3] = X[1] - X[3];

    X[1] = X[7] + X[5];
    X[5] = X[7] - X[5];
    X[7] = X[0] - X[3];
    X[3] = X[0] + X[3];
    X[0] = X[4] + T;
    X[4] = X[4] - T;   
  }
  
  __device__ __forceinline__ void ntt16() {
     
    #pragma unroll
     for(uint32_t i=0;i<4;i++)
       ntt4(X[i], X[i+4], X[i+8], X[i+12]);
    
     X[5] = X[5] * test_scalar::omega4(1);
     X[6] = X[6] * test_scalar::omega4(2);
     X[7] = X[7] * test_scalar::omega4(3);

     X[9] = X[9] * test_scalar::omega4(2);
     X[10] = X[10] * test_scalar::omega4(4);
     X[11] = X[11] * test_scalar::omega4(6);

     X[13] = X[13] * test_scalar::omega4(3);
     X[14] = X[14] * test_scalar::omega4(6);
     X[15] = X[15] * test_scalar::omega4(9);
    
     #pragma unroll
     for(uint32_t i=0;i<16;i+=4)
       ntt4(X[i], X[i+1], X[i+2], X[i+3]);
     
  }

  // __device__ __forceinline__ void ntt16_win8ct2() {
  //   // #pragma unroll
  //   // for (int i = 1; i < 16; i++)
  //   // {
  //   //   // X[i] = X[i] * test_scalar::omega16(((blockIdx.x * blockDim.x + threadIdx.x) >> 8) * i);
  //   //   X[i] = X[i] * test_scalar::omega4(3);
  //   //   // X[i] = X[i] * X[i];
  //   // }
     
  //   #pragma unroll
  //    for(uint32_t i=0;i<2;i++)
  //      ntt8win(X[i], X[i+2], X[i+4], X[i+6], X[i+8], X[i+10], X[i+12], X[i+14]);
    
  //    X[3] = X[3] * test_scalar::omega4(1);
  //    X[5] = X[5] * test_scalar::omega4(2);
  //    X[7] = X[7] * test_scalar::omega4(3);
  //    X[9] = X[9] * test_scalar::omega4(4);
  //    X[11] = X[11] * test_scalar::omega4(5);
  //    X[13] = X[13] * test_scalar::omega4(6);
  //    X[15] = X[15] * test_scalar::omega4(7);
     
  //    #pragma unroll
  //    for(uint32_t i=0;i<16;i+=2)
  //      ntt2(X[i], X[i+1]);
     
  // }

  __device__ __forceinline__ void ntt16win() {
    test_scalar temp;

    // 1
    temp  = X[0] + X[8];
    X[0]  = X[0] - X[8];
    X[8]  = X[4] + X[12];
    X[4]  = X[4] - X[12];
    X[12] = X[2] + X[10];
    X[2]  = X[2] - X[10];
    X[10] = X[6] + X[14];
    X[6]  = X[6] - X[14];
    X[14] = X[1] + X[9];
    X[1]  = X[1] - X[9];
    X[9]  = X[5] + X[13];
    X[5]  = X[5] - X[13];
    X[13] = X[3] + X[11];
    X[3]  = X[3] - X[11];
    X[11] = X[7] + X[15];
    X[7]  = X[7] - X[15];
   
    X[4] = test_scalar::win4(3) * X[4];

    // 2
    X[15] = temp  + X[8];
    temp  = temp  - X[8];
    X[8]  = X[0]  + X[4];
    X[0]  = X[0]  - X[4];
    X[4]  = X[12] + X[10];
    X[12] = X[12] - X[10];
    X[10] = X[2]  + X[6];
    X[2]  = X[2]  - X[6];
    X[6]  = X[14] + X[9];
    X[14] = X[14] - X[9];
    X[9]  = X[13] + X[11];
    X[13] = X[13] - X[11];
    X[11] = X[1]  + X[7];
    X[1]  = X[1]  - X[7];
    X[7]  = X[3]  + X[5];
    X[3]  = X[3]  - X[5];

    X[12] = test_scalar::win4(5) * X[12];
    X[10] = test_scalar::win4(6) * X[10];
    X[2]  = test_scalar::win4(7) * X[2];

    // 3
    X[5]  = X[10] + X[2];
    X[10] = X[10] - X[2];
    X[2]  = X[6]  + X[9];
    X[6]  = X[6]  - X[9];
    X[9]  = X[14] + X[13];
    X[14] = X[14] - X[13];

    X[13] = X[11] + X[7];
    X[13] = test_scalar::win4(14) * X[13];
    X[11] = test_scalar::win4(12) * X[11] + X[13];
    X[7]  = test_scalar::win4(13) * X[7]  + X[13];

    X[13] = X[1] + X[3];
    X[13] = test_scalar::win4(17) * X[13];
    X[1]  = test_scalar::win4(15) * X[1] + X[13];
    X[3]  = test_scalar::win4(16) * X[3] + X[13];

    // 4
    X[13] = X[15] + X[4];
    X[15] = X[15] - X[4];
    X[4]  = temp  + X[12];
    temp  = temp  - X[12];
    X[12] = X[8]  + X[5];
    X[8]  = X[8]  - X[5];
    X[5]  = X[0]  + X[10];
    X[0]  = X[0]  - X[10];

    X[6]   = test_scalar::win4(9)  * X[6];
    X[9]   = test_scalar::win4(10) * X[9];
    X[14]  = test_scalar::win4(11) * X[14];

    X[10] = X[9]  + X[14];
    X[9]  = X[9]  - X[14];
    X[14] = X[11] + X[1];
    X[11] = X[11] - X[1];
    X[1]  = X[7]  + X[3];
    X[7]  = X[7]  - X[3];

    // 5
    X[3]  = X[13] + X[2];
    X[13] = X[13] - X[2];
    X[2]  = X[15] + X[6];
    X[15] = X[15] - X[6];
    X[6]  = X[4]  + X[10];
    X[4]  = X[4]  - X[10];
    X[10] = temp + X[9];
    temp  = temp - X[9];
    X[9]  = X[12] + X[14];
    X[12] = X[12] - X[14];
    X[14] = X[8]  + X[7];
    X[8]  = X[8]  - X[7];
    X[7]  = X[5]  + X[1];
    X[5]  = X[5]  - X[1];
    X[1]  = X[0]  + X[11];
    X[0]  = X[0]  - X[11];

    // reorder + return;
    X[11] = X[0];
    X[0]  = X[3];
    X[3]  = X[7];
    X[7]  = X[1];
    X[1]  = X[9];
    X[9]  = X[12];
    X[12] = X[15];
    X[15] = X[11];
    X[11] = X[5];
    X[5]  = X[14];
    X[14] = temp;
    temp  = X[8];
    X[8]  = X[13];
    X[13] = temp;
    temp  = X[4];
    X[4]  = X[2];
    X[2]  = X[6];
    X[6]  = X[10];
    X[10] = temp;

  }

  
//   __device__ __forceinline__ void ntt16win() {
//     //does not work
     
//     test_scalar    Y[16];

//     Y[0] = X[0] + X[8];
//     Y[1] = X[0] - X[8];
//     Y[2] = X[4] + X[12];
//     Y[3] = X[4] - X[12];

//     Y[4] = X[2] + X[10];
//     Y[6] = X[2] - X[10];
//     Y[5] = X[6] + X[14];
//     Y[7] = X[6] - X[14];
//     Y[0] = Y[0] * test_scalar::win4(0);
//     Y[1] = Y[1] * test_scalar::win4(1);
//     Y[2] = Y[2] * test_scalar::win4(2);
//     Y[3] = Y[3] * test_scalar::win4(3);

//     Y[8] = X[1] + X[9];
//     Y[12] = X[1] - X[9];
//     Y[9] = X[3] + X[11];
//     Y[13] = X[3] - X[11];
//     Y[10] = X[5] + X[13];
//     Y[14] = X[5] - X[13];
//     Y[11] = X[7] + X[15];
//     Y[15] = X[7] - X[15];

// //#######################################

//     X[0] = Y[0] + Y[2];
//     X[2] = Y[0] - Y[2];
//     X[1] = Y[1] + Y[3];
//     X[3] = Y[1] - Y[3];

//     X[4] = Y[4] + Y[5];
//     X[5] = Y[4] - Y[5];
//     X[6] = Y[6] + Y[7];
//     X[7] = Y[6] - Y[7];
//     X[4] = X[4] * test_scalar::win4(4);
//     X[5] = X[5] * test_scalar::win4(5);
//     X[6] = X[6] * test_scalar::win4(6);
//     X[7] = X[7] * test_scalar::win4(7);

//     X[8] = Y[8] + Y[10];
//     X[10] = Y[8] - Y[10];
//     X[9] = Y[9] + Y[11];
//     X[11] = Y[9] - Y[11];

//     X[12] = Y[12] + Y[14];
//     X[14] = Y[12] - Y[14];
//     X[13] = Y[13] + Y[15];
//     X[15] = Y[13] - Y[15];

//     Y[0] = X[6] + X[7];
//     X[7] = X[6] - X[7];
//     X[6] = Y[0];

//     Y[0] = X[8] + X[9];
//     X[9] = X[8] - X[9];
//     X[8] = Y[0];

//     Y[0] = X[10] + X[11];
//     X[11] = X[10] - X[11];
//     X[10] = Y[0];

//     Y[0] = X[12] + X[14];
//     X[14] = X[12] - X[14];
//     X[12] = Y[0];

//     Y[0] = X[13] + X[15];
//     X[15] = X[13] - X[15];
//     X[15] = Y[0];

//     X[8] = X[8] * test_scalar::win4(8);
//     X[9] = X[9] * test_scalar::win4(9);
//     X[10] = X[10] * test_scalar::win4(10);
//     X[11] = X[11] * test_scalar::win4(11);

//     Y[0] = X[12] + X[13];
//     Y[0] = Y[0] * test_scalar::win4(14);
//     X[12] = X[12] * test_scalar::win4(12);
//     X[13] = X[13] * test_scalar::win4(13);
//     X[12] = X[12] + Y[0];
//     X[13] = X[13] + Y[0]; //switch!

//     Y[0] = X[14] + X[15];
//     Y[0] = Y[0] * test_scalar::win4(17);
//     X[14] = X[14] * test_scalar::win4(16);
//     X[15] = X[15] * test_scalar::win4(15);
//     X[14] = X[14] + Y[0];
//     X[15] = X[15] + Y[0];

// //#######################################

//     Y[0] = X[0] + X[4];
//     Y[4] = X[0] - X[4];
//     Y[1] = X[1] + X[6];
//     Y[5] = X[1] - X[6];
//     Y[2] = X[2] + X[5];
//     Y[6] = X[2] - X[5];
//     Y[3] = X[3] + X[7];
//     Y[7] = X[3] - X[7];

//     Y[8] = X[8];
//     Y[9] = X[9];
//     Y[10] = X[10] + X[11];
//     Y[11] = X[10] - X[11];

//     Y[12] = X[13] + X[15];
//     Y[14] = X[13] - X[15];
//     Y[13] = X[12] + X[14];
//     Y[15] = X[12] - X[14];

//     //#################################

//     X[0] = Y[0] + Y[8];
//     X[8] = Y[0] - Y[8];
//     X[1] = Y[1] + Y[9];
//     X[9] = Y[1] - Y[9];
//     X[2] = Y[2] + Y[12];
//     X[10] = Y[2] - Y[12];
//     X[3] = Y[3] + Y[14];
//     X[11] = Y[3] - Y[14];
//     X[4] = Y[4] + Y[10];
//     X[12] = Y[4] - Y[10];
//     X[5] = Y[5] + Y[11];
//     X[13] = Y[5] - Y[11];
//     X[6] = Y[6] + Y[13];
//     X[14] = Y[6] - Y[13];
//     X[7] = Y[7] + Y[15];
//     X[15] = Y[7] - Y[15];
 
//   }

//     __device__ __forceinline__ void ntt16win_lowreg() {
//      //complete nonsense - just for performance test

//     test_scalar    T;

//     T = X[0] + X[8];
//     X[1] = X[0] - X[8];
//     X[2] = X[4] + X[12];
//     X[3] = X[4] - X[12];

//     T = X[2] + X[10];
//     X[6] = X[2] - X[10];
//     X[5] = X[6] + X[14];
//     X[7] = X[6] - T;
//     X[0] = X[0] * test_scalar::win4(0);
//     X[1] = X[1] * test_scalar::win4(1);
//     X[2] = X[2] * test_scalar::win4(2);
//     X[3] = X[3] * test_scalar::win4(3);

//     X[8] = X[1] + T;
//     X[12] = X[1] - X[9];
//     T = X[3] + X[11];
//     X[13] = X[3] - X[11];
//     X[10] = X[5] + X[13];
//     X[14] = X[5] - X[13];
//     X[11] = X[7] + X[15];
//     X[15] = X[7] - X[15];

// //#######################################

//     X[0] = X[0] + X[2];
//     X[2] = X[0] - T;
//     T = X[1] + X[3];
//     X[3] = X[1] - X[3];

//     X[4] = X[4] + X[5];
//     X[5] = X[4] - X[5];
//     X[6] = X[6] + X[7];
//     X[7] = X[6] - T;
//     X[4] = X[4] * test_scalar::win4(4);
//     X[5] = X[5] * test_scalar::win4(5);
//     X[6] = X[6] * test_scalar::win4(6);
//     T = T * test_scalar::win4(7);

//     X[8] = X[8] + X[10];
//     X[10] = X[8] - X[10];
//     X[9] = X[9] + X[11];
//     T = T - X[11];

//     X[12] = X[12] + X[14];
//     X[14] = X[12] - X[14];
//     X[13] = X[13] + X[15];
//     X[15] = X[13] - X[15];

//     X[0] = X[6] + X[7];
//     X[7] = X[6] - X[7];
//     X[6] = X[0];

//     X[0] = X[8] + X[9];
//     X[9] = X[8] - X[9];
//     X[8] = X[0];

//     X[0] = X[10] + X[11];
//     X[11] = X[10] - X[11];
//     X[10] = X[0];

//     X[0] = X[12] + X[14];
//     X[14] = X[12] - X[14];
//     X[12] = X[0];

//     X[0] = X[13] + X[15];
//     X[15] = X[13] - X[15];
//     X[15] = X[0];

//     X[8] = X[8] * test_scalar::win4(8);
//     X[9] = X[9] * test_scalar::win4(9);
//     X[10] = X[10] * test_scalar::win4(10);
//     X[11] = X[11] * test_scalar::win4(11);

//     X[0] = X[12] + X[13];
//     X[0] = X[0] * test_scalar::win4(14);
//     X[12] = X[12] * test_scalar::win4(12);
//     X[13] = X[13] * test_scalar::win4(13);
//     X[12] = X[12] + X[0];
//     X[13] = X[13] + X[0]; //switch!

//     X[0] = X[14] + X[15];
//     X[0] = X[0] * test_scalar::win4(17);
//     X[14] = X[14] * test_scalar::win4(16);
//     X[15] = X[15] * test_scalar::win4(15);
//     X[14] = X[14] + X[0];
//     X[15] = X[15] + X[0];

// //#######################################

//     X[0] = X[0] + X[4];
//     X[4] = X[0] - X[4];
//     X[1] = X[1] + X[6];
//     X[5] = X[1] - X[6];
//     X[2] = X[2] + X[5];
//     X[6] = X[2] - X[5];
//     X[3] = X[3] + X[7];
//     X[7] = X[3] - X[7];

//     X[8] = X[8];
//     X[9] = X[9];
//     X[10] = X[10] + X[11];
//     X[11] = X[10] - X[11];

//     X[12] = X[13] + X[15];
//     X[14] = X[13] - X[15];
//     X[13] = X[12] + X[14];
//     X[15] = X[12] - X[14];

//     //#################################

//     X[0] = X[0] + X[8];
//     X[8] = X[0] - X[8];
//     X[1] = X[1] + X[9];
//     X[9] = X[1] - X[9];
//     X[2] = X[2] + X[12];
//     X[10] = X[2] - X[12];
//     X[3] = X[3] + X[14];
//     X[11] = X[3] - X[14];
//     X[4] = X[4] + X[10];
//     X[12] = X[4] - X[10];
//     X[5] = X[5] + X[11];
//     X[13] = X[5] - X[11];
//     X[6] = X[6] + X[13];
//     X[14] = X[6] - X[13];
//     X[7] = X[7] + X[15];
//     X[15] = X[7] - X[15];
 
//   }
    
  __device__ __forceinline__ void SharedDataColumns(uint4 *shmem, bool store, bool high_bits) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = threadIdx.x & 0x7;
    uint32_t column_id = threadIdx.x >> 3;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    // uint4 temp{0,0,0,0};
    
    #pragma unroll
    for(uint32_t i=0;i<4;i++) {
      #pragma unroll
      for(uint32_t j=0;j<4;j++){
        uint32_t row_id = i*4+j;
        // if (threadIdx.x==0) printf("ntt id %d row id %d col id %d shmem add %d \n", ntt_id, row_id, column_id, ntt_id * 256 + row_id * 16 + column_id);
        if (store) {
          shmem[ntt_id * 256 + row_id * 16 + column_id] = X[i+j*4].load_half(high_bits);           // note transpose here
          // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = X[i+j*4].load_half(high_bits);           // note transpose here
          // temp.w = threadIdx.x;
          // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = temp;           // note transpose here
        }
        else {
          X[i+j*4].store_half(shmem[ntt_id * 256 + row_id * 16 + column_id], high_bits);
          // X[i+j*4].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + column_id], high_bits);
        }
      }
    }
  }

  __device__ __forceinline__ void SharedDataRows(uint4 *shmem, bool store, bool high_bits) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = threadIdx.x & 0x7;
    uint32_t row_id = threadIdx.x >> 3;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    
    #pragma unroll
    for(uint32_t i=0;i<16;i++) {
      if (store) {
        // shmem[ntt_id * 264 + (row_id * 33)/2 + i] = X[i].load_half(high_bits);
        shmem[ntt_id * 256 + row_id * 16 + i] = X[i].load_half(high_bits);
      }
      else {
        // X[i].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + i] ,high_bits);   
        X[i].store_half(shmem[ntt_id * 256 + row_id * 16 + i] ,high_bits);   
      }
    }
  }

  __device__ __forceinline__ void SharedDataColumns2(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0x7 : threadIdx.x >> 3;
    uint32_t column_id = stride? threadIdx.x >> 3 : threadIdx.x & 0x7;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    // uint4 temp{0,0,0,0};
    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      // if (threadIdx.x==0) printf("ntt id %d row id %d col id %d shmem add %d \n", ntt_id, row_id, column_id, ntt_id * 256 + row_id * 16 + column_id);
      if (store) {
        shmem[ntt_id * 64 + i * 8 + column_id] = X[i].load_half(high_bits);           // note transpose here
        // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = X[i+j*4].load_half(high_bits);           // note transpose here
        // temp.w = threadIdx.x;
        // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = temp;           // note transpose here
      }
      else {
        X[i].store_half(shmem[ntt_id * 64 + i * 8 + column_id], high_bits);
        // X[i+j*4].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + column_id], high_bits);
      }
    }
  }

  __device__ __forceinline__ void SharedDataRows2(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0x7 : threadIdx.x >> 3;
    uint32_t row_id = stride? threadIdx.x >> 3 : threadIdx.x & 0x7;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      if (store) {
        // shmem[ntt_id * 264 + (row_id * 33)/2 + i] = X[i].load_half(high_bits);
        shmem[ntt_id * 64 + row_id * 8 + i] = X[i].load_half(high_bits);
      }
      else {
        // X[i].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + i] ,high_bits);   
        X[i].store_half(shmem[ntt_id * 64 + row_id * 8 + i] ,high_bits);   
      }
    }
  }

  __device__ __forceinline__ void SharedData32Columns8(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;
    // if (threadIdx.x!=16) return;

    uint32_t ntt_id = stride? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t column_id = stride? threadIdx.x >> 4 : threadIdx.x & 0x3;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    // uint4 temp{0,0,0,0};
    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      // printf("fgdf %d, ",ntt_id * 32 + i * 4 + column_id);
      // printf("fgdf %d, ",X[i]);
      // if (threadIdx.x==0) printf("ntt id %d row id %d col id %d shmem add %d \n", ntt_id, row_id, column_id, ntt_id * 256 + row_id * 16 + column_id);
      if (store) {
        shmem[ntt_id * 32 + i * 4 + column_id] = X[i].load_half(high_bits);           // note transpose here
        // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = X[i+j*4].load_half(high_bits);           // note transpose here
        // temp.w = threadIdx.x;
        // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = temp;           // note transpose here
      }
      else {
        X[i].store_half(shmem[ntt_id * 32 + i * 4 + column_id], high_bits);
        // X[i+j*4].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + column_id], high_bits);
      }
    }
  }

  __device__ __forceinline__ void SharedData32Rows8(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t row_id = stride? threadIdx.x >> 4 : threadIdx.x & 0x3;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      if (store) {
        // shmem[ntt_id * 264 + (row_id * 33)/2 + i] = X[i].load_half(high_bits);
        shmem[ntt_id * 32 + row_id * 8 + i] = X[i].load_half(high_bits);
      }
      else {
        // X[i].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + i] ,high_bits);   
        X[i].store_half(shmem[ntt_id * 32 + row_id * 8 + i] ,high_bits);   
      }
    }
  }

  __device__ __forceinline__ void SharedData32Columns4_2(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t column_id = (stride? threadIdx.x >> 4 : threadIdx.x & 0x3)*2;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    // uint4 temp{0,0,0,0};
    
    #pragma unroll
    for(uint32_t j=0;j<2;j++) {
      #pragma unroll
      for(uint32_t i=0;i<4;i++) {
        // if (threadIdx.x==0) printf("ntt id %d row id %d col id %d shmem add %d \n", ntt_id, row_id, column_id, ntt_id * 256 + row_id * 16 + column_id);
        if (store) {
          shmem[ntt_id * 32 + i * 8 + column_id + j] = X[4*j+i].load_half(high_bits);           // note transpose here
          // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = X[i+j*4].load_half(high_bits);           // note transpose here
          // temp.w = threadIdx.x;
          // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = temp;           // note transpose here
        }
        else {
          X[4*j+i].store_half(shmem[ntt_id * 32 + i * 8 + column_id + j], high_bits);
          // X[i+j*4].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + column_id], high_bits);
        }
      }
    }
  }

  __device__ __forceinline__ void SharedData32Rows4_2(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0xf : threadIdx.x >> 2;
    uint32_t row_id = (stride? threadIdx.x >> 4 : threadIdx.x & 0x3)*2;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    
    #pragma unroll
    for(uint32_t j=0;j<2;j++) {
      #pragma unroll
      for(uint32_t i=0;i<4;i++) {
        if (store) {
          // shmem[ntt_id * 264 + (row_id * 33)/2 + i] = X[i].load_half(high_bits);
          shmem[ntt_id * 32 + row_id * 4 + 4*j + i] = X[4*j+i].load_half(high_bits);
        }
        else {
          // X[i].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + i] ,high_bits);   
          X[4*j+i].store_half(shmem[ntt_id * 32 + row_id * 4 + 4*j + i] ,high_bits);   
        }
      }
    }
  }




  __device__ __forceinline__ void SharedData16Columns8(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;
    // if (threadIdx.x!=16) return;

    uint32_t ntt_id = stride? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t column_id = stride? threadIdx.x >> 5 : threadIdx.x & 0x1;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    // uint4 temp{0,0,0,0};
    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      // printf("fgdf %d, ",ntt_id * 32 + i * 4 + column_id);
      // printf("fgdf %d, ",X[i]);
      // if (threadIdx.x==0) printf("ntt id %d row id %d col id %d shmem add %d \n", ntt_id, row_id, column_id, ntt_id * 256 + row_id * 16 + column_id);
      if (store) {
        shmem[ntt_id * 16 + i * 2 + column_id] = X[i].load_half(high_bits);           // note transpose here
        // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = X[i+j*4].load_half(high_bits);           // note transpose here
        // temp.w = threadIdx.x;
        // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = temp;           // note transpose here
      }
      else {
        X[i].store_half(shmem[ntt_id * 16 + i * 2 + column_id], high_bits);
        // X[i+j*4].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + column_id], high_bits);
      }
    }
  }

  __device__ __forceinline__ void SharedData16Rows8(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t row_id = stride? threadIdx.x >> 5 : threadIdx.x & 0x1;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    
    #pragma unroll
    for(uint32_t i=0;i<8;i++) {
      if (store) {
        // shmem[ntt_id * 264 + (row_id * 33)/2 + i] = X[i].load_half(high_bits);
        shmem[ntt_id * 16 + row_id * 8 + i] = X[i].load_half(high_bits);
      }
      else {
        // X[i].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + i] ,high_bits);   
        X[i].store_half(shmem[ntt_id * 16 + row_id * 8 + i] ,high_bits);   
      }
    }
  }

  __device__ __forceinline__ void SharedData16Columns2_4(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t column_id = (stride? threadIdx.x >> 5 : threadIdx.x & 0x1)*4;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    // uint4 temp{0,0,0,0};
    
    #pragma unroll
    for(uint32_t j=0;j<4;j++) {
      #pragma unroll
      for(uint32_t i=0;i<2;i++) {
        // if (threadIdx.x==0) printf("ntt id %d row id %d col id %d shmem add %d \n", ntt_id, row_id, column_id, ntt_id * 256 + row_id * 16 + column_id);
        if (store) {
          shmem[ntt_id * 16 + i * 8 + column_id + j] = X[2*j+i].load_half(high_bits);           // note transpose here
          // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = X[i+j*4].load_half(high_bits);           // note transpose here
          // temp.w = threadIdx.x;
          // shmem[ntt_id * 264 + (row_id * 33)/2 + column_id] = temp;           // note transpose here
        }
        else {
          X[2*j+i].store_half(shmem[ntt_id * 16 + i * 8 + column_id + j], high_bits);
          // X[i+j*4].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + column_id], high_bits);
        }
      }
    }
  }

  __device__ __forceinline__ void SharedData16Rows2_4(uint4 *shmem, bool store, bool high_bits, bool stride) {
    // const uint32_t stride=blockDim.x+1;

    uint32_t ntt_id = stride? threadIdx.x & 0x1f : threadIdx.x >> 1;
    uint32_t row_id = (stride? threadIdx.x >> 5 : threadIdx.x & 0x1)*4;
    
    
    // addr=addr + threadIdx.x*8;          // use odd stride to avoid bank conflicts

    // todo - stride to avoid bank conflicts, use ptx commands
    
    #pragma unroll
    for(uint32_t j=0;j<4;j++) {
      #pragma unroll
      for(uint32_t i=0;i<2;i++) {
        if (store) {
          // shmem[ntt_id * 264 + (row_id * 33)/2 + i] = X[i].load_half(high_bits);
          shmem[ntt_id * 16 + row_id * 2 + 2*j + i] = X[2*j+i].load_half(high_bits);
        }
        else {
          // X[i].store_half(shmem[ntt_id * 264 + (row_id * 33)/2 + i] ,high_bits);   
          X[2*j+i].store_half(shmem[ntt_id * 16 + row_id * 2 + 2*j + i] ,high_bits);   
        }
      }
    }
  }


  // __device__ __forceinline__ void load_twiddles(uint32_t ntt8_num) {
  //   #pragma unroll 
  //   for (int i = 1; i < 8; i++)
  //   {
  //     // X[i] = X[i] * test_scalar::omega8(threadIdx.x&0x7);
  //     W[i-1] = test_scalar::omega8(ntt8_num);
  //     // X[i] = X[i] * test_scalar::omega4(i);
  //     // X[i] = X[i] * X[i];
  //   }
  // }

  __device__ __forceinline__ void twiddlesInternal() {
    #pragma unroll 
    for (int i = 1; i < 8; i++)
    {
      // X[i] = X[i] * test_scalar::omega8(threadIdx.x&0x7);
      // if (threadIdx.x==9) printf("xbef %d wbef %d\n", X[i], WE[i]);
      X[i] = X[i] * WI[i-1];
      // if (threadIdx.x==9) printf("xaf %d\n", X[i]);
      // X[i] = X[i] * test_scalar::omega8(i);
      // X[i] = X[i] * X[i];
    }
  }

  // __device__ __forceinline__ void twiddles32() {
  //   #pragma unroll 
  //   for (int i = 1; i < 8; i++)
  //   {
  //     // X[i] = X[i] * test_scalar::omega8(threadIdx.x&0x7);
  //     // if (threadIdx.x==9) printf("xbef %d wbef %d\n", X[i], WE[i]);
  //     X[i] = X[i] * WI[i-1];
  //     // if (threadIdx.x==9) printf("xaf %d\n", X[i]);
  //     // X[i] = X[i] * test_scalar::omega8(i);
  //     // X[i] = X[i] * X[i];
  //   }
  // }

  __device__ __forceinline__ void twiddlesExternal() {
    #pragma unroll 
    for (int i = 0; i < 8; i++)
    {
      // X[i] = X[i] * test_scalar::omega8(threadIdx.x&0x7);
      // if (threadIdx.x==9) printf("xbef %d wbef %d\n", X[i], WE[i]);
      X[i] = X[i] * WE[i];
      // if (threadIdx.x==9) printf("xaf %d\n", X[i]);
      // X[i] = X[i] * test_scalar::omega8(i);
      // X[i] = X[i] * X[i];
    }
  }

   __device__ __forceinline__ void plus() {
    #pragma unroll 
    for (int i = 1; i < 16; i++)
    {
      // X[i] = X[i] * test_scalar::omega16(((blockIdx.x * blockDim.x + threadIdx.x) >> 8) * i);
      X[i] = X[i] + X[i];
      // X[i] = X[i] * X[i];
    }
  }

  // __device__ __forceinline__ void loadSharedDataAndTwiddle32x32(uint32_t addr) {
  //   const uint32_t stride=blockDim.x+1;
  //   uint64_t       currentRoot;
  //   uint64_t       load;
    
  //   currentRoot=1;             // For INTT, we could save the final mult step by initializing this to 1024^-1 mod p
    
  //   addr=addr + (threadIdx.x & 0xFFE0)*8 + (threadIdx.x & 0x1F)*stride*8;   // use odd stride to avoid bank conflicts
    
  //   #pragma unroll 1
  //   for(int32_t i=0;i<4;i++) {
  //     // we can't afford 32 copies of this
  //     #pragma unroll
  //     for(uint32_t j=0;j<8;j++) {
  //       load=load_shared_u64(addr + (i*8 + j)*8);
  //       X[24+j] = Math::mul<false>(load, currentRoot);                // 8 copies of this
  //       currentRoot = mul(currentRoot, threadRoot);                   // 8 copies of this        
  //     }

  //     switch(i) {
  //       case 0: copy8<0, 24>(); break;
  //       case 1: copy8<8, 24>(); break;
  //       case 2: copy8<16, 24>(); break;
  //       case 3: break;
  //     }
  //   }
  // }

};  

#endif