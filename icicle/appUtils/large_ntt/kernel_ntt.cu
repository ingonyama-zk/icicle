#ifndef K_NTT
#define K_NTT
#pragma once


#include "thread_ntt.cu"


__device__ uint32_t dig_rev(uint32_t num, uint32_t log_size, bool dit){
  uint32_t rev_num=0, temp, dig_len;
  if (dit){
    for (int i = 4; i >= 0; i--)
    {
      dig_len = STAGE_SIZES_DEVICE[log_size][i];
      temp = num & ((1<<dig_len)-1);
      num = num >> dig_len;
      rev_num = rev_num << dig_len;
      rev_num = rev_num | temp;
    }
  }
  else{
    for (int i = 0; i < 5; i++)
    {
      dig_len = STAGE_SIZES_DEVICE[log_size][i];
      temp = num & ((1<<dig_len)-1);
      num = num >> dig_len;
      rev_num = rev_num << dig_len;
      rev_num = rev_num | temp;
    }
  }
  return rev_num;
  
}

__launch_bounds__(64)
__global__ void reorder_digits_kernel(uint4* arr, uint4* arr_reordered, uint32_t log_size, bool dit){
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t rd = tid;
  uint32_t wr = dig_rev(tid, log_size, dit);
  arr_reordered[wr] = arr[rd];
  arr_reordered[wr + (1<<log_size)] = arr[rd + (1<<log_size)];
}

__launch_bounds__(64)
__global__ void ntt64(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint4* basic_twiddles, uint32_t log_size, uint32_t data_stride, uint32_t log_data_stride, uint32_t twiddle_stride, bool strided, uint32_t stage_num, bool inv, bool dit) {
  NTTEngine engine;
  stage_metadata s_meta;
  extern __shared__ uint4 shmem[];

  s_meta.th_stride = 8;
  s_meta.ntt_block_size = 64;
  s_meta.ntt_block_id = (blockIdx.x << 3) + (strided? (threadIdx.x & 0x7) : (threadIdx.x >> 3));
  s_meta.ntt_inp_id = strided? (threadIdx.x >> 3) : (threadIdx.x & 0x7);
  
  engine.loadBasicTwiddles(basic_twiddles);
  engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
  if (twiddle_stride && dit) {
    engine.loadExternalTwiddles(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    engine.twiddlesExternal();
  }
  engine.loadInternalTwiddles(internal_twiddles, strided);

  #pragma unroll 1
  for(uint32_t phase=0;phase<2;phase++) {
    
    engine.ntt8win(); 
    if(phase==0) {
      engine.SharedDataColumns2(shmem, true, false, strided); //store low
      __syncthreads();
      engine.SharedDataRows2(shmem, false, false, strided); //load low
      engine.SharedDataRows2(shmem, true, true, strided); //store high
      __syncthreads();
      engine.SharedDataColumns2(shmem, false, true, strided); //load high
      engine.twiddlesInternal();
    }
  }
  
  if (twiddle_stride && !dit) {
    engine.loadExternalTwiddles(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    engine.twiddlesExternal();
  }
  engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
}

__launch_bounds__(64)
__global__ void ntt32(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint4* basic_twiddles, uint32_t log_size, uint32_t data_stride, uint32_t log_data_stride, uint32_t twiddle_stride, bool strided, uint32_t stage_num, bool inv, bool dit) {
  NTTEngine engine;
  stage_metadata s_meta;
  extern __shared__ uint4 shmem[];

  s_meta.th_stride = 4;
  s_meta.ntt_block_size = 32;
  s_meta.ntt_block_id = (blockIdx.x << 4) + (strided? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
  s_meta.ntt_inp_id = strided? (threadIdx.x >> 4) : (threadIdx.x & 0x3);
    
  engine.loadBasicTwiddles(basic_twiddles);
  engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
  engine.loadInternalTwiddles32(internal_twiddles, strided);
  engine.ntt8win(); 
  engine.twiddlesInternal();
  engine.SharedData32Columns8(shmem, true, false, strided); //store low
  __syncthreads();
  engine.SharedData32Rows4_2(shmem, false, false, strided); //load low
  engine.SharedData32Rows8(shmem, true, true, strided); //store high
  __syncthreads();
  engine.SharedData32Columns4_2(shmem, false, true, strided); //load high
  engine.ntt4_2();
  if (twiddle_stride) {
    engine.loadExternalTwiddles32(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    engine.twiddlesExternal();
  }
  engine.storeGlobalData32(out, data_stride, log_data_stride, log_size, strided, s_meta);  
}

__launch_bounds__(64)
__global__ void ntt32dit(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint4* basic_twiddles, uint32_t log_size, uint32_t data_stride, uint32_t log_data_stride, uint32_t twiddle_stride, bool strided, uint32_t stage_num, bool inv, bool dit) {

  NTTEngine engine;
  stage_metadata s_meta;
  extern __shared__ uint4 shmem[];

  s_meta.th_stride = 4;
  s_meta.ntt_block_size = 32;
  s_meta.ntt_block_id = (blockIdx.x << 4) + (strided? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
  s_meta.ntt_inp_id = strided? (threadIdx.x >> 4) : (threadIdx.x & 0x3);
  
  engine.loadBasicTwiddles(basic_twiddles);
  engine.loadGlobalData32(in, data_stride, log_data_stride, log_size, strided, s_meta);
  if (twiddle_stride) {
    engine.loadExternalTwiddles32(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    engine.twiddlesExternal();
  }
  engine.loadInternalTwiddles32(internal_twiddles, strided);
  engine.ntt4_2(); 
  engine.SharedData32Columns4_2(shmem, true, false, strided); //store low
  __syncthreads();
  engine.SharedData32Rows8(shmem, false, false, strided); //load low
  engine.SharedData32Rows4_2(shmem, true, true, strided); //store high
  __syncthreads();
  engine.SharedData32Columns8(shmem, false, true, strided); //load high
  engine.twiddlesInternal();
  engine.ntt8win();
  engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
}

__launch_bounds__(64)
__global__ void ntt16(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint4* basic_twiddles, uint32_t log_size, uint32_t data_stride, uint32_t log_data_stride, uint32_t twiddle_stride, bool strided, uint32_t stage_num, bool inv, bool dit) {

  NTTEngine engine;
  stage_metadata s_meta;
  extern __shared__ uint4 shmem[];

  s_meta.th_stride = 2;
  s_meta.ntt_block_size = 16;
  s_meta.ntt_block_id = (blockIdx.x << 5) + (strided? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
  s_meta.ntt_inp_id = strided? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

  engine.loadBasicTwiddles(basic_twiddles);
  engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
  engine.loadInternalTwiddles16(internal_twiddles, strided);
  engine.ntt8win(); 
  engine.twiddlesInternal();
  engine.SharedData16Columns8(shmem, true, false, strided); //store low
  __syncthreads();
  engine.SharedData16Rows2_4(shmem, false, false, strided); //load low
  engine.SharedData16Rows8(shmem, true, true, strided); //store high
  __syncthreads();
  engine.SharedData16Columns2_4(shmem, false, true, strided); //load high
  engine.ntt2_4();
  if (twiddle_stride) {
    engine.loadExternalTwiddles16(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    engine.twiddlesExternal();
  }
  engine.storeGlobalData16(out, data_stride, log_data_stride, log_size, strided, s_meta);
}

__launch_bounds__(64)
__global__ void ntt16dit(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint4* basic_twiddles, uint32_t log_size, uint32_t data_stride, uint32_t log_data_stride, uint32_t twiddle_stride, bool strided, uint32_t stage_num, bool inv, bool dit) {

  NTTEngine engine;
  stage_metadata s_meta;
  extern __shared__ uint4 shmem[];

  s_meta.th_stride = 2;
  s_meta.ntt_block_size = 16;
  s_meta.ntt_block_id = (blockIdx.x << 5) + (strided? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
  s_meta.ntt_inp_id = strided? (threadIdx.x >> 5) : (threadIdx.x & 0x1);
  
  engine.loadBasicTwiddles(basic_twiddles);
  engine.loadGlobalData16(in, data_stride, log_data_stride, log_size, strided, s_meta);
  if (twiddle_stride) {
    engine.loadExternalTwiddles16(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    engine.twiddlesExternal();
  }
  engine.loadInternalTwiddles16(internal_twiddles, strided);
  engine.ntt2_4(); 
  engine.SharedData16Columns2_4(shmem, true, false, strided); //store low
  __syncthreads();
  engine.SharedData16Rows8(shmem, false, false, strided); //load low
  engine.SharedData16Rows2_4(shmem, true, true, strided); //store high
  __syncthreads();
  engine.SharedData16Columns8(shmem, false, true, strided); //load high
  engine.twiddlesInternal();
  engine.ntt8win();
  engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
}


__global__ void normalize_kernel(uint4* data, uint32_t size, test_scalar norm_factor){
  test_scalar temp;
  temp.store_half(data[threadIdx.x], false);
  temp.store_half(data[threadIdx.x+size], true);
  temp = temp * norm_factor;
  data[threadIdx.x] = temp.load_half(false);
  data[threadIdx.x+size] = temp.load_half(true);
}

__global__ void generate_base_table(test_scalar basic_root, uint4* base_table){
  
  test_scalar w = basic_root;
  test_scalar t = test_scalar::one();
  for (int i = 0; i < 64; i++)
  {
    base_table[i] = t.load_half(false);
    base_table[i+64] = t.load_half(true);
    t = t * w;
  }
}

__global__ void generate_basic_twiddles(test_scalar basic_root, uint4* basic_twiddles){
  
  test_scalar w0 = basic_root*basic_root;
  test_scalar w1 = (basic_root + w0*basic_root)*test_scalar::inv_log_size(1);
  test_scalar w2 = (basic_root - w0*basic_root)*test_scalar::inv_log_size(1);
  basic_twiddles[0] = w0.load_half(false);
  basic_twiddles[3] = w0.load_half(true);
  basic_twiddles[1] = w1.load_half(false);
  basic_twiddles[4] = w1.load_half(true);
  basic_twiddles[2] = w2.load_half(false);
  basic_twiddles[5] = w2.load_half(true);
}

__global__ void generate_twiddle_combinations(uint4* w6_table, uint4* w12_table, uint4* w18_table, uint4* w24_table, uint4* w30_table, uint4* twiddles, uint32_t log_size, uint32_t stage_num, test_scalar norm_factor){
  
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t range1 = 0, range2 = 0, ind;
  for (ind = 0; ind < stage_num; ind++) range1 += STAGE_SIZES_DEVICE[log_size][ind];
  range2 = STAGE_SIZES_DEVICE[log_size][ind];
  uint32_t root_order = range1 + range2;
  uint32_t exp = ((tid & ((1<<range1)-1)) * (tid >> range1)) << (30-root_order);
  test_scalar w6,w12,w18,w24,w30;
  w6.store_half(w6_table[exp >> 24], false);
  w6.store_half(w6_table[(exp >> 24) + 64], true);
  w12.store_half(w12_table[((exp >> 18) & 0x3f)], false);
  w12.store_half(w12_table[((exp >> 18) & 0x3f) + 64], true);
  w18.store_half(w18_table[((exp >> 12) & 0x3f)], false);
  w18.store_half(w18_table[((exp >> 12) & 0x3f) + 64], true);
  w24.store_half(w24_table[((exp >> 6) & 0x3f)], false);
  w24.store_half(w24_table[((exp >> 6) & 0x3f) + 64], true);
  w30.store_half(w30_table[(exp & 0x3f)], false);
  w30.store_half(w30_table[(exp & 0x3f) + 64], true);
  test_scalar t = w6 * w12 * w18 * w24 * w30 * norm_factor;
  twiddles[tid + LOW_W_OFFSETS[log_size][stage_num]] = t.load_half(false);
  twiddles[tid + HIGH_W_OFFSETS[log_size][stage_num]] = t.load_half(true);
}

uint4* generate_external_twiddles(test_scalar basic_root, uint4* twiddles, uint4* basic_twiddles, uint32_t log_size, bool inv){

  uint4* w6_table;
  uint4* w12_table;
  uint4* w18_table;
  uint4* w24_table;
  uint4* w30_table;
  cudaMalloc((void**)&w6_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w12_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w18_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w24_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w30_table, sizeof(uint4)*64*2);

  test_scalar temp_root = basic_root;
  generate_base_table<<<1,1>>>(basic_root, w30_table);
  for (int i=0; i<6; i++) temp_root = temp_root*temp_root;
  generate_base_table<<<1,1>>>(temp_root, w24_table);
  for (int i=0; i<6; i++) temp_root = temp_root*temp_root;
  generate_base_table<<<1,1>>>(temp_root, w18_table);
  for (int i=0; i<6; i++) temp_root = temp_root*temp_root;
  generate_base_table<<<1,1>>>(temp_root, w12_table);
  for (int i=0; i<6; i++) temp_root = temp_root*temp_root;
  generate_base_table<<<1,1>>>(temp_root, w6_table);
  for (int i=0; i<3; i++) temp_root = temp_root*temp_root;
  generate_basic_twiddles<<<1,1>>>(temp_root, basic_twiddles);

  uint32_t temp = STAGE_SIZES_HOST[log_size][0];
  for (int i = 1; i < 5; i++)
  {
    if (!STAGE_SIZES_HOST[log_size][i]) break;
    temp += STAGE_SIZES_HOST[log_size][i];
    generate_twiddle_combinations<<<1<<(temp-8),256>>>(w6_table, w12_table, w18_table, w24_table, w30_table, twiddles, log_size, i, (temp == log_size && inv)? test_scalar::inv_log_size(log_size) : test_scalar::one()); 
  }
  return w6_table;
}

void new_ntt(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint4* basic_twiddles, uint32_t log_size, bool inv, bool dit){

  if (log_size==4){
    if (dit) {ntt16dit<<<1,4,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);}
    else {ntt16<<<1,4,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);}
    if (inv) normalize_kernel<<<1,16>>>(out, 16, test_scalar::inv_log_size(4));
    return;
  }
  if (log_size==5){
    if (dit) {ntt32dit<<<1,4,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);}
    else {ntt32<<<1,4,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);}
    if (inv) normalize_kernel<<<1,32>>>(out, 32, test_scalar::inv_log_size(5));
    return;
  }
  if (log_size==6){
    ntt64<<<1,8,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
    if (inv) normalize_kernel<<<1,64>>>(out, 64, test_scalar::inv_log_size(6));
    return;
  }
  if (log_size==8){
    ntt16<<<1,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 16, 4, 16, true, 1, inv, dit); //we need threads 32+ although 16-31 are idle
    ntt16<<<1,32,8*64*sizeof(uint4)>>>(out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
    return;
  }
  
  if (dit){
    for (int i = 0; i < 5; i++)
    {
      uint32_t stage_size = STAGE_SIZES_HOST[log_size][i];
      uint32_t stride_log = 0;
      for (int j=0; j<i; j++) stride_log += STAGE_SIZES_HOST[log_size][j];
      if (stage_size == 6) ntt64<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(i? out: in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1<<stride_log, stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
      if (stage_size == 5) ntt32dit<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(i? out: in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1<<stride_log, stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
      if (stage_size == 4) ntt16dit<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(i? out: in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1<<stride_log, stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
    }
  }
  else{
    bool first_run = false, prev_stage = false;
    for (int i = 4; i >= 0; i--)
    {
      uint32_t stage_size = STAGE_SIZES_HOST[log_size][i];
      uint32_t stride_log = 0;
      for (int j=0; j<i; j++) stride_log += STAGE_SIZES_HOST[log_size][j];
      first_run = stage_size && !prev_stage;
      if (stage_size == 6) ntt64<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(first_run? in : out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1<<stride_log, stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
      if (stage_size == 5) ntt32<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(first_run? in : out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1<<stride_log, stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
      if (stage_size == 4) ntt16<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(first_run? in : out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1<<stride_log, stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
      prev_stage = stage_size;
    }
  }
}

#endif