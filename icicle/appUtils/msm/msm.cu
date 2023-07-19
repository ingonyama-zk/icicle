#ifndef MSM
#define MSM
#pragma once
#include <stdexcept>
#include <cuda.h>
#include <cooperative_groups.h>
#include "../../primitives/affine.cuh"
#include <iostream>
#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/device_ptr.h>
// #include <thrust/algorithm.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include "../../utils/cuda_utils.cuh"
#include "../../primitives/projective.cuh"
#include "../../primitives/field.cuh"
#include "msm.cuh"

#define TEMP_NUM 10
#define MAX_TH 256
#define MAX_BUCKET_SIZE 9

// #define SIGNED_DIG
// #define BIG_TRIANGLE
// #define ZPRIZE
// #define SSM_SUM  //WIP
// #define PHASE1_TEST

#define SIZE 32
#define SHMEM_SIZE 64 * 4 //why this size?

// For last iteration (saves useless work)
// Use volatile to prevent caching in registers (compiler optimization)
// No __syncthreads() necessary!
template <typename P>
__device__ void warpReduce(P* shmem_ptr, int t, int first, int last) {
  for (int i=first; i>last; i>>=1){
    shmem_ptr[t] = shmem_ptr[t] + shmem_ptr[t + i];
  }
}

template <typename P>
__global__ void general_sum_reduction_kernel(P *v, P *v_r, unsigned nof_partial_sums, unsigned write_stride, unsigned write_phase) {
	// Allocate shared memory
	__shared__ P partial_sum[SHMEM_SIZE]; //use memory allocation like coop groups
	// int partial_sum[];

	// Calculate thread ID
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	for (int s = blockDim.x / 2; s > nof_partial_sums-1; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
  //todo - add device function
	// if (threadIdx.x < 32) {
	// 	warpReduce(partial_sum, threadIdx.x);
	// }

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x < nof_partial_sums) {
    unsigned write_ind = nof_partial_sums*blockIdx.x + threadIdx.x;
		v_r[((write_ind/write_stride)*2 + write_phase)*write_stride + write_ind%write_stride] = partial_sum[threadIdx.x];
	}
}

template <typename P>
__global__ void single_stage_multi_reduction_kernel(P *v, P *v_r, unsigned block_size, unsigned write_stride, unsigned write_phase, unsigned padding) {
	// Allocate shared memory
	// __shared__ P partial_sum[SHMEM_SIZE]; //use memory allocation like coop groups
	// int partial_sum[];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_p = padding? (tid/(2*padding))*padding + tid%padding: tid;
  int jump =block_size/2;
  int block_id = tid_p/jump;
  int block_tid = tid_p%jump;

  // if (block_tid < jump){
  unsigned read_ind = block_size*block_id + block_tid; //fix
  // unsigned padded_read_ind = block_size*block_id + block_tid; //fix
  // unsigned write_ind = jump*block_id + block_tid;
  unsigned write_ind = tid;
  if (padding) printf(" %u %u %u %u\n",tid,tid_p,read_ind,((write_ind/write_stride)*2 + write_phase)*write_stride + write_ind%write_stride);
	v_r[write_stride? ((write_ind/write_stride)*2 + write_phase)*write_stride + write_ind%write_stride : write_ind] = padding? (tid%(2*padding)<padding)? v[read_ind] + v[read_ind + jump] : P::zero() :v[read_ind] + v[read_ind + jump];
  // }
}

template <typename P>
__global__ void variable_block_multi_reduction_kernel(P *v, P *v_r, unsigned *block_sizes, unsigned *block_offsets, unsigned write_stride, unsigned write_phase) {
	// Allocate shared memory
	// __shared__ P partial_sum[SHMEM_SIZE]; //use memory allocation like coop groups
	// int partial_sum[];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int jump =block_sizes[tid]/2; //????
  int block_offset = block_offsets[tid]; //block 
  int block_tid = tid - block_offset/2; //fix

  // if (block_tid < jump){
  unsigned read_ind = block_offset + block_tid; //fix
  // unsigned padded_read_ind = block_size*block_id + block_tid; //fix
  // unsigned write_ind = jump*block_id + block_tid;
  unsigned write_ind = block_offset/2 + block_tid;
  // if (padding) printf(" %u %u %u %u\n",tid,tid_p,read_ind,((write_ind/write_stride)*2 + write_phase)*write_stride + write_ind%write_stride);
	v_r[write_stride? ((write_ind/write_stride)*2 + write_phase)*write_stride + write_ind%write_stride : write_ind] = v[read_ind] + v[read_ind + jump];
  // }
}

template <typename P>
__global__ void pad_buckets_kernel(P *v, P *v_r, unsigned block_size) {
	// Allocate shared memory
	// __shared__ P partial_sum[SHMEM_SIZE]; //use memory allocation like coop groups
	// int partial_sum[];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int dont_write = (tid/block_size)%2;

  v_r[tid] = dont_write? P::zero() : v[(tid/(block_size*2))*block_size + tid%block_size];
}


template <typename P> //todo-add SM and device function
__global__ void reduce_triangles_kernel(P *source_buckets,P* temp_buckets, P *target_buckets, const unsigned source_c, const unsigned source_nof_bms) {
	// Allocate shared memory
	// __shared__ int partial_sum[SHMEM_SIZE];
	
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned source_nof_buckets = source_nof_bms<<source_c;//2*2^8
  // if (tid ==0) printf("source_nof_buckets %u\n",source_nof_buckets);
  // if (tid ==9) printf("dims %u %u %u\n",blockIdx.x,blockDim.x,threadIdx.x);
  const unsigned source_nof_bm_buckets = 1<<source_c;//2^8
  unsigned temp_nof_bm_buckets = source_nof_bm_buckets;//2^8
  const unsigned target_nof_bms = source_nof_bms<<1;//4
  const unsigned target_c = source_c>>1;//4
  // const unsigned target_nof_buckets = target_nof_bms<<target_c;
  const unsigned target_nof_bm_buckets = 1<<target_c;//2^4
  unsigned nof_threads_per_bm = source_nof_bm_buckets>>1;//2^7
  // unsigned nof_threads_per_bm = target_nof_bm_buckets>>1;
  // if (tid >= source_nof_buckets>>1) return; //total threads
  unsigned bm_index = tid/nof_threads_per_bm; //blockidx
  unsigned bm_bucket_index = tid%nof_threads_per_bm; //threadidx
  unsigned bucket_index = bm_index*source_nof_bm_buckets + bm_bucket_index;

  // if (tid ==0) printf("source_nof_buckets %u\n",source_nof_buckets);
  // if (tid ==0) printf("source_nof_bm_buckets %u\n",source_nof_bm_buckets);
  // if (tid ==0) printf("temp_nof_bm_buckets %u\n",temp_nof_bm_buckets);
  // if (tid ==0) printf("target_nof_bms %u\n",target_nof_bms);
  // if (tid ==0) printf("target_c %u\n",target_c);
  // if (tid ==0) printf("target_nof_bm_buckets %u\n",target_nof_bm_buckets);
  // if (tid ==0) printf("nof_threads_per_bm %u\n",nof_threads_per_bm);
	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	// int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  // __syncthreads();
  // if (tid ==0){ 
  //   // printf("t\n");
  //   // for (int i=0;i<source_nof_bms;i++){
  //   // for (int j=0;j<source_nof_bm_buckets;j++)
  //   // {printf("%u ",source_buckets[i*source_nof_bm_buckets+j].x.x);}
  //   // printf("\n");
  //   // }
  //   printf("t\n");
  //   for (int i=0;i<TEMP_NUM;i++)
  //   {printf("%u ",source_buckets[i]);}
  //   printf("\n");
  //   }
  // if (tid ==0) printf("\n");
    // __syncthreads();
	// Store first partial result instead of just the elements
	temp_buckets[bucket_index] = source_buckets[bucket_index] + source_buckets[bucket_index + nof_threads_per_bm];
  // cooperative_groups::grid_group g = cooperative_groups::this_grid(); 
  // g.sync();
	__syncthreads();
  // if (tid ==32) printf("tid %u bucket_index %u temp_buckets[tid] %u\n",tid,bucket_index,temp_buckets[tid].x.x);

  // if (tid ==0){ 
  //   // for (int i=0;i<source_nof_bms<<1;i++){
  //   // for (int j=0;j<source_nof_bm_buckets>>1;j++)
  //   // {printf("%u ",temp_buckets[i*(source_nof_bm_buckets>>1)+j].x.x);}
  //   // printf("\n");
  //   // }
  //   for (int i=0;i<TEMP_NUM;i++)
  //   {printf("%u ",temp_buckets[i]);}
  //   printf("\n");
  //   }
	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	// for (int s = blockDim.x / 2; s > 32; s >>= 1) {
	for (int s = nof_threads_per_bm/2; s > target_nof_bm_buckets/2; s >>= 1) {
		// Each thread does work unless it is further than the stride
    // temp_nof_bm_buckets = temp_nof_bm_buckets>>1;
    // nof_threads_per_bm = temp_nof_bm_buckets>>1;
    // bm_index = tid/nof_threads_per_bm;
    // bm_bucket_index = tid%nof_threads_per_bm;
    // bucket_index = bm_index*source_nof_bm_buckets + bm_bucket_index;
    // if (tid ==9) printf("inds %u %u %u\n",bm_index,bm_bucket_index,bucket_index);
		// if (tid < source_nof_bms*s) {
    if (threadIdx.x < s) {
			temp_buckets[bucket_index] = temp_buckets[bucket_index] + temp_buckets[bucket_index + s];
		}
		__syncthreads();
    // if (tid ==0){ 
    //   for (int i=0;i<TEMP_NUM;i++)
    //    {printf("%u ",temp_buckets[i]);}
    //    printf("\n");
    //   }
	}


	// if (bm_bucket_index < 32) {
	// 	warpReduce(temp_buckets, bucket_index, min(32,nof_threads_per_bm/2), target_nof_bm_buckets/2);
	// }

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	// if (tid < source_nof_bms*target_nof_bm_buckets) { //optimize - last calculation needs to write too
	if (threadIdx.x < target_nof_bm_buckets) { //optimize - last calculation needs to write too
    // if (tid ==9) printf("inds %u %u %u\n",bm_index*temp_nof_bm_buckets + bm_bucket_index,bucket_index);
		target_buckets[bm_index*target_nof_bm_buckets*2 + bm_bucket_index] = temp_buckets[bucket_index];
    // if (bm_index*target_nof_bm_buckets*2 + bm_bucket_index==0) printf("tidddddd %u\n",temp_buckets[bucket_index].x.x);
	}
  // if (tid ==0){ 
  //   for (int i=0;i<TEMP_NUM;i++)
  //    {printf("%u ",target_buckets[i]);}
  //    printf("\n");
  //   }
}

template <typename P>
__global__ void reduce_rectangles_kernel(P *source_buckets,P* temp_buckets, P *target_buckets, const unsigned source_c, const unsigned source_nof_bms) {
	// Allocate shared memory
	// __shared__ int partial_sum[SHMEM_SIZE];
	
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned source_nof_buckets = source_nof_bms<<source_c;
  unsigned source_nof_bm_buckets = 1<<source_c;
  // const unsigned target_nof_bms = source_nof_bms<<1;
  const unsigned target_c = source_c>>1;
  // const unsigned target_nof_buckets = target_nof_bms<<target_c;
  const unsigned source_nof_segment_buckets = source_nof_bm_buckets>>target_c;
  unsigned temp_nof_segment_buckets = source_nof_segment_buckets;
  unsigned target_nof_bm_buckets = 1<<target_c; //==segments per bm
  // unsigned temp_nof_bm_buckets = 1<<target_c;
  unsigned nof_threads_per_bm = source_nof_bm_buckets>>1;//2^7
  unsigned nof_threads_per_segment = source_nof_segment_buckets>>1; //difference between kernels
  // if (tid >= source_nof_buckets>>1) return; //total threads
  unsigned bm_index = tid/nof_threads_per_bm; //blockidx
  unsigned bm_bucket_index = tid%nof_threads_per_bm; //threadidx
  unsigned segment_index = bm_bucket_index/nof_threads_per_segment;
  unsigned segment_bucket_index = bm_bucket_index%nof_threads_per_segment;
  unsigned bucket_index = bm_index*source_nof_bm_buckets + segment_index*source_nof_segment_buckets + segment_bucket_index;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	// int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
  // if (tid ==0){ 
  //   printf("rtar\n");
  //   for (int i=0;i<TEMP_NUM;i++)
  //   {printf("%u ",target_buckets[i]);}
  //   printf("\n");
  //   }
	temp_buckets[bucket_index] = source_buckets[bucket_index] + source_buckets[bucket_index + nof_threads_per_segment];
	__syncthreads();

  // if (tid ==0){ 
  //   for (int i=0;i<TEMP_NUM;i++)
  //   {printf("%u ",temp_buckets[i]);}
  //   printf("\n");
  //   }
	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	// for (int s = blockDim.x / 2; s > 32; s >>= 1) {
	for (int s = nof_threads_per_segment/2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
    // temp_nof_segment_buckets = temp_nof_segment_buckets>>1;
    // nof_threads_per_segment = temp_nof_segment_buckets>>1;
    // segment_index = tid/nof_threads_per_segment;
    // segment_bucket_index = tid%nof_threads_per_segment;
    // bucket_index = segment_index*source_nof_segment_buckets + segment_bucket_index;
		// if (tid < source_nof_bms*target_nof_bm_buckets*s) { //nof segments per bm
		if (segment_bucket_index < s) { //nof segments per bm
			temp_buckets[bucket_index] = temp_buckets[bucket_index] + temp_buckets[bucket_index + s];
		}
		__syncthreads();
    // if (tid ==0){ 
    //   for (int i=0;i<TEMP_NUM;i++)
    //    {printf("%u ",temp_buckets[i]);}
    //    printf("\n");
    //   }
	}


	// if (bm_bucket_index < 32) {
	// 	warpReduce(temp_buckets, bucket_index, min(32,nof_threads_per_bm/2), target_nof_bm_buckets/2);
	// }

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block

	// if (tid < source_nof_bms*target_nof_bm_buckets) {
	if (segment_bucket_index == 0) {
    unsigned src_idx = bm_index*source_nof_bm_buckets + segment_index*source_nof_segment_buckets + segment_bucket_index;
    unsigned dst_idx = target_nof_bm_buckets*(1+bm_index*2) + segment_index;
    target_buckets[dst_idx] = temp_buckets[src_idx];
    // if (dst_idx==0) printf("tirrrr %u %u\n\n\n\n\n",tid,temp_buckets[src_idx]);
    // printf("tid %u dst_idx %u\n",tid,dst_idx);
    // segment_index = tid/target_nof_bm_buckets;
    // segment_bucket_index = tid%target_nof_bm_buckets;
    // bucket_index = target_nof_bm_buckets + segment_index*target_nof_bm_buckets*2 + segment_bucket_index;
		// target_buckets[bucket_index] = temp_buckets[target_nof_bm_buckets*tid];
	}
  // if (tid ==0){ 
  //   for (int i=0;i<TEMP_NUM;i++)
  //    {printf("%u ",target_buckets[i]);}
  //    printf("\n");
  //   }
}

unsigned log2_floor(const unsigned value) {
  unsigned v = value;
  unsigned result = 0;
  while (v >>= 1)
    result++;
  return result;
}

unsigned log2_ceiling(const unsigned value) { return value <= 1 ? 0 : log2_floor(value - 1) + 1; }

unsigned get_optimal_log_data_split(const unsigned mpc, const unsigned source_window_bits, const unsigned target_window_bits,
  const unsigned target_windows_count) {
#define MAX_THREADS 32
#define MIN_BLOCKS 12
const unsigned full_occupancy = mpc * MAX_THREADS * MIN_BLOCKS;
const unsigned target = full_occupancy << 6;
const unsigned unit_threads_count = target_windows_count << target_window_bits;
const unsigned split_target = log2_ceiling(target / unit_threads_count);
const unsigned split_limit = source_window_bits - target_window_bits - 1;
return std::min(split_target, split_limit);
}

template <typename T>
static constexpr __device__ __forceinline__ T ld_single(const T *ptr) {
return __ldg(ptr);
};

template <class T, typename U, unsigned STRIDE>
static constexpr __device__ __forceinline__ T ld(const T *address, const unsigned offset) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % sizeof(U) == 0);
  constexpr size_t count = sizeof(T) / sizeof(U);
  T result = {};
  auto pa = reinterpret_cast<const U *>(address) + offset;
  auto pr = reinterpret_cast<U *>(&result);
#pragma unroll
  for (unsigned i = 0; i < count; i++) {
    const auto pai = pa + i * STRIDE;
    const auto pri = pr + i;
    *pri = ld_single<U>(pai);
  }
  return result;
}

template <class T, unsigned STRIDE = 1, typename U = std::enable_if_t<sizeof(T) % sizeof(uint4) == 0, uint4>>
static constexpr __device__ __forceinline__ T memory_load(const T *address, const unsigned offset = 0, [[maybe_unused]] uint4 _dummy = {}) {
  return ld<T, U, STRIDE>(address, offset);
};

template <class T, unsigned STRIDE = 1, typename U = std::enable_if_t<(sizeof(T) % sizeof(uint4) != 0) && (sizeof(T) % sizeof(uint2) == 0), uint2>>
static constexpr __device__ __forceinline__ T memory_load(const T *address, const unsigned offset = 0, [[maybe_unused]] uint2 _dummy = {}) {
  return ld<T, U, STRIDE>(address, offset);
};

template <class T, unsigned STRIDE = 1, typename U = std::enable_if_t<sizeof(T) % sizeof(uint2) != 0, unsigned>>
static constexpr __device__ __forceinline__ T memory_load(const T *address, const unsigned offset = 0, [[maybe_unused]] unsigned _dummy = {}) {
  return ld<T, U, STRIDE>(address, offset);
};

//this kernel performs single scalar multiplication
//each thread multilies a single scalar and point
template <typename P, typename S>
__global__ void ssm_kernel(S *scalars, P *points, P *results, unsigned N) {

  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) results[tid] = scalars[tid]*points[tid];

}

//this kernel sums all the elements in a given vector using multiple threads
template <typename P>
__global__ void sum_reduction_kernel(P *v, P* v_r) {

	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Start at 1/2 block stride and divide by two each iteration
	for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			v[tid] = v[tid] + v[tid + s];
		}
    __syncthreads();
	}

	// Let the thread 0 for this block write the final result
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = v[tid];
	}
}

//this kernel initializes the buckets with zero points
//each thread initializes a different bucket
template <typename P>
__global__ void initialize_buckets_kernel(P *buckets, unsigned N) {
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) buckets[tid] = P::zero(); //zero point

}

//this kernel splits the scalars into digits of size c
//each thread splits a single scalar into nof_bms digits
template <typename S>
__global__ void split_scalars_kernel(unsigned *buckets_indices, unsigned *point_indices, S *scalars, unsigned total_size, unsigned msm_log_size, unsigned nof_bms, unsigned bm_bitsize, unsigned c, unsigned top_bm_nof_missing_bits){
  
  constexpr unsigned sign_mask = 0x80000000;
  // constexpr unsigned trash_bucket = 0x80000000;
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned bucket_index;
  unsigned bucket_index2;
  unsigned current_index;
  unsigned msm_index = tid >> msm_log_size;
  unsigned borrow = 0;
  if (tid < total_size){
    S scalar = scalars[tid];
    // A point = points[tid];
    // if (scalar == S::zero() || scalar == S::one()) return;
    // if (scalar == S::zero()) return;
    // if (tid == 0) printf("scalar %u", scalar);

    for (unsigned bm = 0; bm < nof_bms; bm++)
    {
      // bucket_index = scalar.get_scalar_digit(bm, c) + (bm==nof_bms-1? ((tid&top_bm_nof_missing_bits)<<(c-top_bm_nof_missing_bits)) : 0);
      bucket_index = scalar.get_scalar_digit(bm, c);
      #ifdef SIGNED_DIG
      bucket_index += borrow;
      borrow = 0;
      unsigned sign = 0;
      // if (tid == 0) printf("index %u", bucket_index);
      if (bucket_index > (1<<(c-1))) {
        bucket_index = (1 << c) - bucket_index;
        borrow = 1;
        sign = sign_mask;
      }
      #endif
      // if (tid == 0) printf("new index %u", bucket_index);
      // if (bm==nof_bms-1) {
      //   bucket_index2 = bucket_index + ((tid&((1<<top_bm_nof_missing_bits)-1))<<(c-top_bm_nof_missing_bits));
      //   if (tid<10) printf("tid %u bi1 %u bi2 %u\n",tid, bucket_index, bucket_index2);
      //   bucket_index = bucket_index2;
      // }
      current_index = bm * total_size + tid;
      #ifdef SIGNED_DIG
      // buckets_indices[current_index] = (msm_index<<(c-1+bm_bitsize)) | (bm<<(c-1)) | bucket_index;  //the bucket module number and the msm number are appended at the msbs
      point_indices[current_index] = sign | tid; //the point index is saved for later
      #else
      buckets_indices[current_index] = (msm_index<<(c+bm_bitsize)) | (bm<<c) | bucket_index;  //the bucket module number and the msm number are appended at the msbs
      if (scalar == S::zero() || scalar == S::one() || bucket_index==0) buckets_indices[current_index] = 0; //will be skipped
      point_indices[current_index] = tid; //the point index is saved for later
      #endif
    }
  }
}

template <typename P, typename A, typename S>
__global__ void add_ones_kernel(A *points, S* scalars, P* results, const unsigned msm_size, const unsigned run_length){
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned nof_threads = (msm_size + run_length - 1)/run_length; //129256
  if (tid>=nof_threads) {
    results[tid] = P::zero();
    return;
  }
  const unsigned start_index = tid*run_length;
  P sum = P::zero();
  for (int i=start_index;i<min(start_index+run_length,msm_size);i++){
    if (scalars[i] == S::one()) sum = sum + points[i];
  }
  results[tid] = sum;
}

__global__ void find_cutoff_kernel(unsigned *v, unsigned size, unsigned cutoff, unsigned run_length, unsigned *result){
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned nof_threads = (size + run_length - 1)/run_length;
  if (tid>=nof_threads) {
    return;
  }
  const unsigned start_index = tid*run_length;
  for (int i=start_index;i<min(start_index+run_length,size-1);i++){
    if (tid<5) printf("index %u size %u\n",i,v[i]);
    if (v[i] > cutoff && v[i+1] <= cutoff) {
      result[0] = i+1;
      printf("result found %u\n",result[0]);
      return;
    }
    if (i == size - 1) {
      result[0] = 0;
      printf("result not found %u\n",result[0]);
    }
  }
}

__global__ void find_max_size(unsigned *bucket_sizes,unsigned *single_bucket_indices,unsigned c, unsigned *largest_bucket_size){
  for (int i=0;;i++){
    if (single_bucket_indices[i]&((1<<c)-1)){
      largest_bucket_size[0] = bucket_sizes[i];
      largest_bucket_size[1] = i;
      break;
    }
  }
}


//this kernel adds up the points in each bucket
// __global__ void accumulate_buckets_kernel(P *__restrict__ buckets, unsigned *__restrict__ bucket_offsets,
  //  unsigned *__restrict__ bucket_sizes, unsigned *__restrict__ single_bucket_indices, unsigned *__restrict__ point_indices, A *__restrict__ points, unsigned nof_buckets, unsigned batch_size, unsigned msm_idx_shift){
template <typename P, typename A>
__global__ void accumulate_buckets_kernel(P *__restrict__ buckets, unsigned *__restrict__ bucket_offsets, unsigned *__restrict__ bucket_sizes, unsigned *__restrict__ single_bucket_indices, const unsigned *__restrict__ point_indices, A *__restrict__ points, const unsigned nof_buckets, const unsigned nof_buckets_to_compute, const unsigned msm_idx_shift, const unsigned c){
  
  constexpr unsigned sign_mask = 0x80000000;
  // constexpr unsigned trash_bucket = 0x80000000;
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  // if (tid>=*nof_buckets_to_compute || tid<11){ 
  if (tid==0) printf("nof_buckets_to_compute %u\n",nof_buckets_to_compute);
  if (tid>=nof_buckets_to_compute){ 
    return;
  }
  if ((single_bucket_indices[tid]&((1<<c)-1))==0)
  {
    // printf("cond %u %u\n",tid,single_bucket_indices[tid]);
    return; //skip zero buckets
  } 
  #ifdef SIGNED_DIG //todo - fix
  const unsigned msm_index = single_bucket_indices[tid]>>msm_idx_shift;
  const unsigned bm_index = (single_bucket_indices[tid]&((1<<msm_idx_shift)-1))>>c;
  const unsigned bucket_index = msm_index * nof_buckets + bm_index * ((1<<(c-1))+1) + (single_bucket_indices[tid]&((1<<c)-1));
  #else
  unsigned msm_index = single_bucket_indices[tid]>>msm_idx_shift;
  unsigned bucket_index = msm_index * nof_buckets + (single_bucket_indices[tid]&((1<<msm_idx_shift)-1));
  #endif
  const unsigned bucket_offset = bucket_offsets[tid];
  const unsigned bucket_size = bucket_sizes[tid];
  // if (bucket_size > MAX_BUCKET_SIZE) {
  //   bucket_offsets[tid] = bucket_offset + MAX_BUCKET_SIZE;
  //   bucket_sizes[tid] = bucket_sizes[tid] - MAX_BUCKET_SIZE;
  // }
  // else single_bucket_indices[tid] = 0;
  // if (bucket_size == 0) {printf("watt"); return;}
  // if (bucket_size > 10) {printf(">10: %u %u %u\n",tid,single_bucket_indices[tid],single_bucket_indices[tid]&((1<<c)-1));}
  if (tid<10) printf("tid %u single_bucket_indices[tid] %u size %u\n", tid, single_bucket_indices[tid],bucket_size);
  // if (tid>=nof_buckets_to_compute-10) printf("tid %u size %u\n", tid, bucket_sizes[tid]);
  // if (tid==0) return;
  // if ((bucket_index>>20)==13) return;
  // if (bucket_sizes[tid]==16777216) printf("tid %u size %u bucket %u offset %u\n", tid, bucket_sizes[tid], bucket_index, bucket_offset);
  // const unsigned *indexes = point_indices + bucket_offset;
  // P bucket = P::zero(); //todo: get rid of init buckets? no.. because what about buckets with no points
  P bucket; //todo: get rid of init buckets? no.. because what about buckets with no points
  // unsigned point_ind;
  // for (unsigned i = 0; i < min(bucket_size,MAX_BUCKET_SIZE); i++)  //add the relevant points starting from the relevant offset up to the bucket size
  for (unsigned i = 0; i < bucket_size; i++)  //add the relevant points starting from the relevant offset up to the bucket size
  {
    // unsigned point_ind = *indexes++;
    // auto point = memory_load<A>(points + point_ind);
    // point_ind = point_indices[bucket_offset+i];
    // bucket = bucket + P::one();
    unsigned point_ind = point_indices[bucket_offset+i];
    #ifdef SIGNED_DIG
    unsigned sign = point_ind & sign_mask;
    point_ind &= ~sign_mask;
    // printf("tid %u sign %u point ind %u \n", tid,sign, point_ind);
    A point = points[point_ind];
    if (sign) point = A::neg(point);
    #else
    A point = points[point_ind];
    #endif
    bucket = i? bucket + point : P::from_affine(point);
    // const unsigned* pa = reinterpret_cast<const unsigned*>(points[point_ind]);
    // P point;
    // Dummy_Scalar scal;
    // scal.x = __ldg(pa);
    // point.x = scal;
    // bucket = bucket + point;
  }
  // buckets[tid] = bucket;
  buckets[bucket_index] = bucket;
}

template <typename P, typename A>
__global__ void accumulate_large_buckets_kernel(P *__restrict__ buckets, unsigned *__restrict__ bucket_offsets, unsigned *__restrict__ bucket_sizes, unsigned *__restrict__ single_bucket_indices, const unsigned *__restrict__ point_indices, A *__restrict__ points, const unsigned nof_buckets, const unsigned nof_buckets_to_compute, const unsigned msm_idx_shift, const unsigned c, const unsigned threads_per_bucket, const unsigned max_run_length){
  
  constexpr unsigned sign_mask = 0x80000000;
  // constexpr unsigned trash_bucket = 0x80000000;
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned large_bucket_index = tid/threads_per_bucket;
  unsigned bucket_segment_index = tid%threads_per_bucket;
  // if (tid>=*nof_buckets_to_compute || tid<11){ 
  if (tid==0) printf("nof_buckets_to_compute large%u\n",nof_buckets_to_compute);
  if (tid==0) printf("nof_buckets_to_compute large2%u\n",nof_buckets_to_compute*threads_per_bucket);
  if (tid>=nof_buckets_to_compute*threads_per_bucket){ 
    return;
  }
  if ((single_bucket_indices[large_bucket_index]&((1<<c)-1))==0) //dont need
  {
    // printf("cond large %u %u\n",tid,single_bucket_indices[large_bucket_index]);
    return; //skip zero buckets
  } 
  // unsigned msm_index = single_bucket_indices[l_tid]>>msm_idx_shift;
  unsigned write_bucket_index = bucket_segment_index * nof_buckets_to_compute + large_bucket_index;
  const unsigned bucket_offset = bucket_offsets[large_bucket_index] + bucket_segment_index*max_run_length;
  // const unsigned bucket_size = max(bucket_sizes[large_bucket_index] - bucket_segment_index*max_run_length,0);
  const unsigned bucket_size = bucket_sizes[large_bucket_index] > bucket_segment_index*max_run_length? bucket_sizes[large_bucket_index] - bucket_segment_index*max_run_length :0;
  // if (bucket_size > MAX_BUCKET_SIZE) {
  //   bucket_offsets[tid] = bucket_offset + MAX_BUCKET_SIZE;
  //   bucket_sizes[tid] = bucket_sizes[tid] - MAX_BUCKET_SIZE;
  // }
  // else single_bucket_indices[tid] = 0;
  // if (bucket_size == 0) {printf("watt"); return;}
  // if (bucket_size > 10) {printf(">10: %u %u %u\n",tid,single_bucket_indices[tid],single_bucket_indices[tid]&((1<<c)-1));}
  // if (tid<50) printf("large tid %u single_bucket_indices[tid] %u size %u\n", tid, single_bucket_indices[large_bucket_index],bucket_size);
  // if (tid>=nof_buckets_to_compute-10) printf("tid %u size %u\n", tid, bucket_sizes[tid]);
  // if (tid==0) return;
  // if ((bucket_index>>20)==13) return;
  // if (bucket_sizes[tid]==16777216) printf("tid %u size %u bucket %u offset %u\n", tid, bucket_sizes[tid], bucket_index, bucket_offset);
  // const unsigned *indexes = point_indices + bucket_offset;
  // P bucket = P::zero(); //todo: get rid of init buckets? no.. because what about buckets with no points
  P bucket; //todo: get rid of init buckets? no.. because what about buckets with no points
  // unsigned point_ind;
  // for (unsigned i = 0; i < min(bucket_size,MAX_BUCKET_SIZE); i++)  //add the relevant points starting from the relevant offset up to the bucket size
  unsigned run_length = min(bucket_size,max_run_length);
  if (tid<8 and tid >3){
    // printf("largeyiugiuvy tid %u run_length %u\n", tid, run_length);
    // printf("largeyiugiuvy bucket_size %u \n", bucket_size);
    // printf("largeyiugiuvy bucket_segment_index*max_run_length %u \n", bucket_segment_index*max_run_length);
    // printf("largeyiugiuvy bucket_sizes[large_bucket_index] %u \n", bucket_sizes[large_bucket_index]);
  }
  for (unsigned i = 0; i < run_length; i++)  //add the relevant points starting from the relevant offset up to the bucket size
  {
    // unsigned point_ind = *indexes++;
    // auto point = memory_load<A>(points + point_ind);
    // point_ind = point_indices[bucket_offset+i];
    // bucket = bucket + P::one();
    unsigned point_ind = point_indices[bucket_offset+i];
    A point = points[point_ind];
    bucket = i? bucket + point : P::from_affine(point);
    // const unsigned* pa = reinterpret_cast<const unsigned*>(points[point_ind]);
    // P point;
    // Dummy_Scalar scal;
    // scal.x = __ldg(pa);
    // point.x = scal;
    // bucket = bucket + point;
  }
  // buckets[tid] = bucket;
  buckets[write_bucket_index] = run_length? bucket : P::zero();
  // printf("large tid %u run_length %u bucket_index %u bucket %u\n", tid, run_length,write_bucket_index,bucket);
}

template <typename P>
__global__ void distribute_large_buckets_kernel(P* large_buckets, P* buckets, unsigned *single_bucket_indices, unsigned size){

  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>=size){ 
    return;
  }
  buckets[single_bucket_indices[tid]] = large_buckets[tid];
}

template <typename P, typename A, typename S>
__global__ void accumulate_buckets_kernel2(P *buckets, A *points, S *scalars, const unsigned c,const unsigned nof_bms, const unsigned size){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>=size) return;
  
  S scalar = scalars[tid];
  A point = points[tid];
  unsigned bucket_index;

  for (unsigned bm = 0; bm < nof_bms; bm++)
  {
    // bucket_index = scalar.get_scalar_digit(bm, c) + (bm==nof_bms-1? ((tid&top_bm_nof_missing_bits)<<(c-top_bm_nof_missing_bits)) : 0);
    bucket_index = scalar.get_scalar_digit(bm, c);
    buckets[bucket_index] = buckets[bucket_index] + point;
  }

}

//this kernel sums the entire bucket module
//each thread deals with a single bucket module
template <typename P>
__global__ void big_triangle_sum_kernel(P* buckets, P* final_sums, unsigned nof_bms, unsigned c){

  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>=nof_bms) return;
  #ifdef SIGNED_DIG
  unsigned buckets_in_bm = (1<<c)+1;
  #else
  unsigned buckets_in_bm = (1<<c);
  #endif
  P line_sum = buckets[(tid+1)*buckets_in_bm-1];
  final_sums[tid] = line_sum;
  for (unsigned i = buckets_in_bm-2; i >0; i--)
  {
    line_sum = line_sum + buckets[tid*buckets_in_bm + i];  //using the running sum method
    final_sums[tid] = final_sums[tid] + line_sum;
  }
}

template <typename P>
__global__ void split_windows_kernel_inner(const unsigned source_window_bits_count, const unsigned source_windows_count,
  const P *__restrict__ source_buckets, P *__restrict__ target_buckets, const unsigned count) {
const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
if (gid >= count) //0,1,2,2^8,2^8+1,2^8+2,32*2^8,32*2^8+1,32*2^8+2^8,32*2^8+2^8+1
return;
const unsigned target_window_bits_count = (source_window_bits_count + 1) >> 1; //8
const unsigned target_windows_count = source_windows_count << 1; //32
const unsigned target_partition_buckets_count = target_windows_count << target_window_bits_count; // 32*2^8
const unsigned target_partitions_count = count / target_partition_buckets_count; //2^7
const unsigned target_partition_index = gid / target_partition_buckets_count; //*0,0,0,0,0,0,1,1,1,1
const unsigned target_partition_tid = gid % target_partition_buckets_count; //*0,1,2,2^8,2^8+1,2^8+2,0,1,2^8,2^8+1
const unsigned target_window_buckets_count = 1 << target_window_bits_count; // 2^8
const unsigned target_window_index = target_partition_tid / target_window_buckets_count; //* 0,0.0,1,1,1,0,0,1,1
const unsigned target_window_tid = target_partition_tid % target_window_buckets_count; //* 0,1,2,0,1,2,0,1,0,1,2
const unsigned split_index = target_window_index & 1; //*0,0,0,1,1,1,0,0,1,1,1
const unsigned source_window_buckets_per_target = source_window_bits_count & 1 // is c odd?
? split_index ? (target_window_tid >> (target_window_bits_count - 1) ? 0 : target_window_buckets_count) //is the target odd?
             : 1 << (source_window_bits_count - target_window_bits_count)
: target_window_buckets_count; //2^8
const unsigned source_window_index = target_window_index >> 1; //*0,0,0,0,0,0,0,0,0,0,0
const unsigned source_offset = source_window_index << source_window_bits_count; //*0,0,0,0,0,0,0,0,0,0,
const unsigned target_shift = target_window_bits_count * split_index; //*0,0,0,8,8,8,0,0,8,8,8
const unsigned target_offset = target_window_tid << target_shift;//*0,1,2,0,2^8,2^9,0,1,0,2^8,2*2^8
const unsigned global_offset = source_offset + target_offset;//*0,1,2,0,2^8,2^9,0,1
const unsigned index_mask = (1 << target_shift) - 1; //*0,0,0,2^8-1,2^8-1,2^8-1,0,0,2^8-1,2^8-1
P target_bucket = P::zero();
#pragma unroll 1
for (unsigned i = target_partition_index; i < source_window_buckets_per_target; i += target_partitions_count) { //from the partition start(*0,0,0,0,0,0,1,1,1,1), stride 2^7, until 2^8 = loop twice
const unsigned index_offset = i & index_mask | (i & ~index_mask) << target_window_bits_count; //*0 2^15,0 2^15,0 2^15,0 2^15,0 2^15,0 2^15,2^8 2^8+2^15,2^8 2^8+2^15,2^8 2^8+2^15,2^8 2^8+2^15
const unsigned load_offset = global_offset + index_offset;//*0 2^15,1 2^15+1,2 2^15+2, 0 2^15, 2^8 2^8+2^15, 2^8 2^8+2^15, 2^8+1 2^8+2^15+1
const auto source_bucket = source_buckets[load_offset];
target_bucket = i == target_partition_index ? source_bucket : target_bucket + source_bucket; //*0+2^15,1+2^15+1,2+2^15+2,...2^8-1+2^15+2^8-1| 0+2^7, 2^8+2^8+2^7...||2^8+2^8+2^15, 2^8+1+2^8+2^15+1...2^9-1+2^9-1+2^15|1+2^7+1, 2^8+1+2^8+2^7+1...
}
target_buckets[gid] = target_bucket; //0,1,2^8,2^8+1,32*2^8,32*2^8+1
}

template <typename P>
__global__ void reduce_buckets_kernel(P *buckets, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  // buckets += gid;
  const auto a = buckets[gid];
  const auto b = buckets[gid+count];
  const P result = a+b;
  buckets[gid] = result;
}

template <typename P>
__global__ void reduce_buckets_kernel2(P *source, P *target, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const auto a = source[gid];
  const auto b = source[gid+count];
  const P result = a+b;
  target[gid] = result;
}

template <typename P>
__global__ void last_pass_gather_kernel(const unsigned bits_count_pass_one, const P *__restrict__ source, P *__restrict__ target,
  const unsigned count) {
const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
if (gid >= count)
return;
unsigned window_index = gid / bits_count_pass_one;
unsigned window_tid = gid % bits_count_pass_one;
for (unsigned bits_count = bits_count_pass_one; bits_count > 1;) {
bits_count = (bits_count + 1) >> 1;
window_index <<= 1;
if (window_tid >= bits_count) {
window_index++;
window_tid -= bits_count;
}
}
const unsigned sid = (window_index << 1) + 1;
const auto pz = source[sid];
// const point_jacobian pj = point_xyzz::to_jacobian(pz, f);
target[gid] = pz;
}

//this kernel uses single scalar multiplication to multiply each bucket by its index
//each thread deals with a single bucket
template <typename P, typename S>
__global__ void ssm_buckets_kernel(P* buckets, unsigned* single_bucket_indices, unsigned nof_buckets, unsigned c){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>nof_buckets) return;
  unsigned bucket_index = single_bucket_indices[tid];
  S scalar_bucket_multiplier;
  scalar_bucket_multiplier = {bucket_index&((1<<c)-1), 0, 0, 0, 0, 0, 0, 0}; //the index without the bucket module index
  buckets[bucket_index] = scalar_bucket_multiplier*buckets[bucket_index];

}

template <typename P>
__global__ void last_pass_kernel(P*final_buckets, P*final_sums, unsigned num_sums){
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>num_sums) return;
  final_sums[tid] = final_buckets[2*tid+1];
}

//this kernel computes the final result using the double and add algorithm
//it is done by a single thread
template <typename P, typename S>
__global__ void final_accumulation_kernel(P* final_sums, P* ones_result, P* final_results, unsigned nof_msms, unsigned nof_bms, unsigned c){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>nof_msms) return;
  P final_result = P::zero();
  for (unsigned i = nof_bms; i >1; i--)
  {
    final_result = final_result + final_sums[i-1 + tid*nof_bms];  //add
    for (unsigned j=0; j<c; j++)  //double
    {
      final_result = final_result + final_result;
    }
  }
  final_results[tid] = final_result + final_sums[tid*nof_bms] + ones_result[0];
  // final_results[tid] = final_result + final_sums[tid*nof_bms];

}

template <typename P>
void test_reduce_triangle(P* h_buckets){
  for (int i=0; i<TEMP_NUM; i++) std::cout<<h_buckets[i]<<" ";
  std::cout<<std::endl;
  P*buckets;
  P*temp;
  P*target;
  unsigned count = TEMP_NUM;
  cudaMalloc(&buckets, sizeof(P) * count);
  cudaMemcpy(buckets, h_buckets, sizeof(P) * count, cudaMemcpyHostToDevice);
  cudaMalloc(&temp, sizeof(P) * count);
  cudaMalloc(&target, sizeof(P) * count);
  // reduce_triangles_kernel<<<4,8>>>(buckets,temp,target,4,4);
  // reduce_triangles_kernel<<<5,8>>>(buckets,temp,target,4,4);
  general_sum_reduction_kernel<<<5,8>>>(buckets,target,4,4,0);
  cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
  
  std::vector<P> h_target;
  h_target.reserve(TEMP_NUM);
  cudaMemcpy(h_target.data(), target, sizeof(P) * TEMP_NUM, cudaMemcpyDeviceToHost);
    std::cout<<cudaGetLastError()<<std::endl;
  std::cout<<"target"<<std::endl;
  for (int i = 0; i < TEMP_NUM; i++)
  {
    std::cout<<h_target[i]<<" ";
  }
  std::cout<<std::endl;

  // std::vector<P> h_buckets;
  // h_buckets.reserve(nof_buckets);
  //   cudaMemcpy(h_buckets.data(), buckets, sizeof(P) * nof_buckets, cudaMemcpyDeviceToHost);
  //   std::cout<<"buckets accumulated"<<std::endl;
  //   for (unsigned i = 0; i < nof_buckets; i++)
  //   {
  //     std::cout<<h_buckets[i]<<" ";
  //   }
  //   std::cout<<std::endl;

}

template <typename P>
void test_reduce_var(P* h_buckets){
  for (int i=0; i<TEMP_NUM; i++) std::cout<<h_buckets[i]<<" ";
  std::cout<<std::endl;
  P*buckets;
  P*temp;
  P*target;
  unsigned count = TEMP_NUM;
  cudaMalloc(&buckets, sizeof(P) * count);
  cudaMemcpy(buckets, h_buckets, sizeof(P) * count, cudaMemcpyHostToDevice);
  cudaMalloc(&temp, sizeof(P) * count);
  cudaMalloc(&target, sizeof(P) * count);
  // reduce_rectangles_kernel<<<5,8>>>(buckets,temp,target,4,4);
  // single_stage_multi_reduction_kernel<<<1,64>>>(buckets,target,16,8,0);
  unsigned h_sizes[10] = {4,4,4,4};
  unsigned h_offsets[10] = {2,2,6,6};
  unsigned *sizes;
  unsigned *offsets;
  cudaMalloc(&sizes, sizeof(unsigned) * count);
  cudaMalloc(&offsets, sizeof(unsigned) * count);
  cudaMemcpy(sizes, h_sizes, sizeof(unsigned) * count, cudaMemcpyHostToDevice);
  cudaMemcpy(offsets, h_offsets, sizeof(unsigned) * count, cudaMemcpyHostToDevice);
  variable_block_multi_reduction_kernel<<<1,4>>>(buckets,target,sizes,offsets,0,0);
  
  cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
  std::vector<P> h_target;
  h_target.reserve(TEMP_NUM);
  cudaMemcpy(h_target.data(), target, sizeof(P) * TEMP_NUM, cudaMemcpyDeviceToHost);
    std::cout<<cudaGetLastError()<<std::endl;
  std::cout<<"target"<<std::endl;
  for (int i = 0; i < TEMP_NUM; i++)
  {
    std::cout<<h_target[i]<<" ";
  }
  std::cout<<std::endl;
}


template <typename P>
void test_reduce_single(P* h_buckets){
  for (int i=0; i<TEMP_NUM; i++) std::cout<<h_buckets[i]<<" ";
  std::cout<<std::endl;
  P*buckets;
  P*temp;
  P*target;
  unsigned count = TEMP_NUM;
  cudaMalloc(&buckets, sizeof(P) * count);
  cudaMemcpy(buckets, h_buckets, sizeof(P) * count, cudaMemcpyHostToDevice);
  cudaMalloc(&temp, sizeof(P) * count);
  cudaMalloc(&target, sizeof(P) * count);
  // reduce_rectangles_kernel<<<5,8>>>(buckets,temp,target,4,4);
  // single_stage_multi_reduction_kernel<<<1,64>>>(buckets,target,16,8,0);
  single_stage_multi_reduction_kernel<<<2,32>>>(buckets,target,2,0,0);
  
  cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
  std::vector<P> h_target;
  h_target.reserve(TEMP_NUM);
  cudaMemcpy(h_target.data(), target, sizeof(P) * TEMP_NUM, cudaMemcpyDeviceToHost);
    std::cout<<cudaGetLastError()<<std::endl;
  std::cout<<"target"<<std::endl;
  for (int i = 0; i < TEMP_NUM; i++)
  {
    std::cout<<h_target[i]<<" ";
  }
  std::cout<<std::endl;
}

template <typename P>
void test_reduce_rectangle(P* h_buckets){
  for (int i=0; i<TEMP_NUM; i++) std::cout<<h_buckets[i]<<" ";
  std::cout<<std::endl;
  P*buckets;
  P*temp;
  P*target;
  unsigned count = TEMP_NUM;
  cudaMalloc(&buckets, sizeof(P) * count);
  cudaMemcpy(buckets, h_buckets, sizeof(P) * count, cudaMemcpyHostToDevice);
  cudaMalloc(&temp, sizeof(P) * count);
  cudaMalloc(&target, sizeof(P) * count);
  // reduce_rectangles_kernel<<<5,8>>>(buckets,temp,target,4,4);
  general_sum_reduction_kernel<<<20,2>>>(buckets,target,1,4,1);
  
  cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
  std::vector<P> h_target;
  h_target.reserve(TEMP_NUM);
  cudaMemcpy(h_target.data(), target, sizeof(P) * TEMP_NUM, cudaMemcpyDeviceToHost);
    std::cout<<cudaGetLastError()<<std::endl;
  std::cout<<"target"<<std::endl;
  for (int i = 0; i < TEMP_NUM; i++)
  {
    std::cout<<h_target[i]<<" ";
  }
  std::cout<<std::endl;
}


//this function computes msm using the bucket method
template <typename S, typename P, typename A>
void bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned size, P* final_result, bool on_device, bool big_triangle, cudaStream_t stream) {
  
  // std::cout<<"points"<<std::endl;
  // for (int i = 0; i < size; i++)
  // {
  //   std::cout<<points[i]<<" ";
  // }
  // std::cout<<std::endl;
  // std::cout<<"scalars"<<std::endl;
  // for (int i = 0; i < size; i++)
  // {
  //   std::cout<<scalars[i]<<" ";
  // }
  // std::cout<<std::endl;

  S *d_scalars;
  A *d_points;
  if (!on_device) {
    //copy scalars and point to gpu
    cudaMallocAsync(&d_scalars, sizeof(S) * size, stream);
    cudaMallocAsync(&d_points, sizeof(A) * size, stream);
    cudaMemcpyAsync(d_scalars, scalars, sizeof(S) * size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_points, points, sizeof(A) * size, cudaMemcpyHostToDevice, stream);
  }
  else {
    d_scalars = scalars;
    d_points = points;
  }

  P *buckets;
  //compute number of bucket modules and number of buckets in each module
  unsigned nof_bms = bitsize/c;
  unsigned msm_log_size = ceil(log2(size));
  unsigned bm_bitsize = ceil(log2(nof_bms));
  if (bitsize%c){
    nof_bms++;
  }
  unsigned top_bm_nof_missing_bits = c*nof_bms - bitsize;
  std::cout << "top_bm_nof_missing_bits" << top_bm_nof_missing_bits <<std::endl;
  #ifdef SIGNED_DIG
  unsigned nof_buckets = nof_bms*((1<<(c-1))+1); //signed digits
  #else
  unsigned nof_buckets = nof_bms<<c;
  #endif
  cudaMallocAsync(&buckets, sizeof(P) * nof_buckets, stream);

  // launch the bucket initialization kernel with maximum threads
  unsigned NUM_THREADS = 1 << 10;
  unsigned NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, nof_buckets);
  // cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());

  //accumulate ones
  P *ones_results; //fix whole division, in last run in kernel too
  // const unsigned nof_runs = max(1<<(msm_log_size-6),16);
  const unsigned nof_runs = msm_log_size > 10? (1<<(msm_log_size-6)) : 16;
  const unsigned run_length = (size + nof_runs -1)/nof_runs;
  cudaMallocAsync(&ones_results, sizeof(P) * nof_runs, stream);
  NUM_THREADS = min(1 << 8,nof_runs);
  NUM_BLOCKS = (nof_runs + NUM_THREADS - 1) / NUM_THREADS;
  add_ones_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(d_points, d_scalars, ones_results, size, run_length);
  // cudaDeviceSynchronize();
  // printf("cuda error ones  %u\n",cudaGetLastError());

  // cudaDeviceSynchronize();
  // std::vector<P> h_ones_results;
  // h_ones_results.reserve(nof_runs);
  // cudaMemcpy(h_ones_results.data(), ones_results, sizeof(P) * nof_runs, cudaMemcpyDeviceToHost);
  // std::cout<<"one results"<<std::endl;
  // for (int i = 0; i < nof_runs; i++)
  // {
  //   std::cout<<h_ones_results[i]<<" ";
  // }
  // std::cout<<std::endl;

  for (int s=nof_runs>>1;s>0;s>>=1){
  NUM_THREADS = min(MAX_TH,s);
  NUM_BLOCKS = (s + NUM_THREADS - 1) / NUM_THREADS;
  single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(ones_results,ones_results,s*2,0,0,0);
  // cudaDeviceSynchronize();
  // printf("cuda error ones  %u\n",cudaGetLastError());
  // cudaDeviceSynchronize();
  // cudaMemcpy(h_ones_results.data(), ones_results, sizeof(P) * nof_runs, cudaMemcpyDeviceToHost);
  // std::cout<<"one results"<<std::endl;
  // for (int i = 0; i < nof_runs; i++)
  // {
  //   std::cout<<h_ones_results[i]<<" ";
  // }
  // std::cout<<std::endl;
  }
  #ifndef PHASE1_TEST

  unsigned *bucket_indices;
  unsigned *point_indices;
  cudaMallocAsync(&bucket_indices, sizeof(unsigned) * size * (nof_bms+1), stream);
  cudaMallocAsync(&point_indices, sizeof(unsigned) * size * (nof_bms+1), stream);

  //split scalars into digits
  NUM_THREADS = 1 << 10;
  NUM_BLOCKS = (size * (nof_bms+1) + NUM_THREADS - 1) / NUM_THREADS;
  split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(bucket_indices + size, point_indices + size, d_scalars, size, msm_log_size, 
                                                    nof_bms, bm_bitsize, c, top_bm_nof_missing_bits); //+size - leaving the first bm free for the out of place sort later
                                                    // cudaDeviceSynchronize();
                                                    printf("cuda error %u\n",cudaGetLastError());


  // cudaDeviceSynchronize();
  // std::vector<unsigned> h_bucket_ind;
  // std::vector<unsigned> h_point_ind;
  // h_bucket_ind.reserve(size * (nof_bms+1));
  // h_point_ind.reserve(size * (nof_bms+1));
  // cudaMemcpy(h_bucket_ind.data(), bucket_indices, sizeof(unsigned) * size * (nof_bms+1), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_point_ind.data(), point_indices, sizeof(unsigned) * size * (nof_bms+1), cudaMemcpyDeviceToHost);
  //   std::cout<<cudaGetLastError()<<std::endl;
  // std::cout<<"buckets inds"<<std::endl;
  // for (int i = 0; i < size * (nof_bms+1); i++)
  // {
  //   std::cout<<h_bucket_ind[i]<<" ";
  // }
  // std::cout<<std::endl;
  // std::cout<<"points inds"<<std::endl;
  // for (int i = 0; i < size * (nof_bms+1); i++)
  // {
  //   std::cout<<h_point_ind[i]<<" ";
  // }
  // std::cout<<std::endl;

  // std::cout<<"pure buckets inds"<<std::endl;
  // for (int i = 0; i < size * (nof_bms+1); i++)
  // {
  //   std::cout<<h_bucket_ind[i]%(1<<(c-1))<<" ";
  // }
  // std::cout<<std::endl;
  // std::cout<<"pure points inds"<<std::endl;
  // for (int i = 0; i < size * (nof_bms+1); i++)
  // {
  //   std::cout<<h_point_ind[i]%(1<<31)<<" ";
  // }
  // std::cout<<std::endl;
                                                    

  //sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
  unsigned *sort_indices_temp_storage{};
  size_t sort_indices_temp_storage_bytes;
  // The second to last parameter is the default value supplied explicitly to allow passing the stream
  // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for more info
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + size, bucket_indices,
                                 point_indices + size, point_indices, size, 0, sizeof(unsigned) * 8, stream);
  cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream);
  for (unsigned i = 0; i < nof_bms; i++) {
    unsigned offset_out = i * size;
    unsigned offset_in = offset_out + size;
    // The second to last parameter is the default value supplied explicitly to allow passing the stream
    // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for more info
    cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in, bucket_indices + offset_out,
                                 point_indices + offset_in, point_indices + offset_out, size, 0, sizeof(unsigned) * 8, stream);
  }
  cudaFreeAsync(sort_indices_temp_storage, stream);

  //find bucket_sizes
  unsigned *single_bucket_indices;
  unsigned *bucket_sizes;
  unsigned *nof_buckets_to_compute;
  cudaMallocAsync(&single_bucket_indices, sizeof(unsigned)*nof_buckets, stream);
  cudaMallocAsync(&bucket_sizes, sizeof(unsigned)*nof_buckets, stream);
  cudaMallocAsync(&nof_buckets_to_compute, sizeof(unsigned), stream);
  unsigned *encode_temp_storage{};
  size_t encode_temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, nof_bms*size, stream);
  cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream);
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, nof_bms*size, stream);
  cudaFreeAsync(encode_temp_storage, stream);

// cudaDeviceSynchronize();
    std::vector<unsigned> h_single;
    std::vector<unsigned> h_sizes;
    h_single.reserve(nof_buckets);
    h_sizes.reserve(nof_buckets);
//     cudaMemcpy(h_single.data(), single_bucket_indices, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_sizes.data(), bucket_sizes, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
//     std::cout<<"single buckets"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_single[i]<<" ";
//     }
//     std::cout<<std::endl;
//     std::cout<<"bucket sizes"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_sizes[i]<<" ";
//     }
//     std::cout<<std::endl;


  //get offsets - where does each new bucket begin
  unsigned* bucket_offsets;
  cudaMallocAsync(&bucket_offsets, sizeof(unsigned)*nof_buckets, stream);
  unsigned* offsets_temp_storage{};
  size_t offsets_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets, stream);
  cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream);
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets, stream);
  cudaFreeAsync(offsets_temp_storage, stream);

  // cudaDeviceSynchronize();
  //   std::vector<unsigned> h_offsets;
  //   h_offsets.reserve(nof_buckets);
  //   cudaMemcpy(h_offsets.data(), bucket_offsets, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
  //   std::cout<<"bucket_offsets"<<std::endl;
  //   for (unsigned i = 0; i < nof_buckets; i++)
  //   {
  //     std::cout<<h_offsets[i]<<" ";
  //   }
  //   std::cout<<std::endl;


  //sort by bucket sizes
  unsigned h_nof_buckets_to_compute;
  cudaMemcpyAsync(&h_nof_buckets_to_compute, nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream);

  unsigned* sorted_bucket_sizes;
  cudaMallocAsync(&sorted_bucket_sizes, sizeof(unsigned)*h_nof_buckets_to_compute, stream);
  unsigned* sorted_bucket_offsets;
  cudaMallocAsync(&sorted_bucket_offsets, sizeof(unsigned)*h_nof_buckets_to_compute, stream);
  unsigned* sort_offsets_temp_storage{};
  size_t sort_offsets_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, bucket_offsets, sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
  cudaMallocAsync(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, stream);
  cub::DeviceRadixSort::SortPairsDescending(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, bucket_offsets, sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
  cudaFreeAsync(sort_offsets_temp_storage, stream);
      
      
  unsigned* sorted_single_bucket_indices;
  cudaMallocAsync(&sorted_single_bucket_indices, sizeof(unsigned)*h_nof_buckets_to_compute, stream);
  unsigned* sort_single_temp_storage{};
  size_t sort_single_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, single_bucket_indices, sorted_single_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
  cudaMallocAsync(&sort_single_temp_storage, sort_single_temp_storage_bytes, stream);
  cub::DeviceRadixSort::SortPairsDescending(sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, single_bucket_indices, sorted_single_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
  cudaFreeAsync(sort_single_temp_storage, stream);

cudaDeviceSynchronize();
cudaMemcpy(h_single.data(), sorted_single_bucket_indices, sizeof(unsigned) * h_nof_buckets_to_compute, cudaMemcpyDeviceToHost);
cudaMemcpy(h_sizes.data(), sorted_bucket_sizes, sizeof(unsigned) * h_nof_buckets_to_compute, cudaMemcpyDeviceToHost);
// cudaMemcpy(h_offsets.data(), bucket_offsets, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
    // std::cout<<"bucket_offsets"<<std::endl;
    // for (unsigned i = 0; i < h_nof_buckets_to_compute; i++)
    // {
    //   std::cout<<h_offsets[i]<<" ";
    // }
    // std::cout<<std::endl;
// std::cout<<"single buckets"<<std::endl;
// for (unsigned i = 0; i < h_nof_buckets_to_compute; i++)
// {
//   std::cout<<h_single[i]<<" ";
// }
// std::cout<<std::endl;
// std::cout<<"bucket sizes"<<std::endl;
// for (unsigned i = 0; i < h_nof_buckets_to_compute; i++)
// {
//   std::cout<<h_sizes[i]<<" ";
// }
// std::cout<<std::endl;

  //find large buckets
  unsigned avarage_size = size/(1<<c);
  printf("avarage_size %u\n", avarage_size);
  float large_bucket_factor = 10; //global param
  unsigned bucket_th = ceil(large_bucket_factor*avarage_size);
  // unsigned bucket_th = 95;
  printf("bucket_th %u\n", bucket_th);

  unsigned *nof_large_buckets;
  cudaMallocAsync(&nof_large_buckets, sizeof(unsigned), stream);

  unsigned TOTAL_THREADS = 129000; //fix - device dependant
  unsigned cutoff_run_length = max(2,h_nof_buckets_to_compute/TOTAL_THREADS);
  unsigned cutoff_nof_runs = (h_nof_buckets_to_compute + cutoff_run_length -1)/cutoff_run_length;
  printf("cutoff_run_length %u\n", cutoff_run_length);
  printf("cutoff_nof_runs %u\n", cutoff_nof_runs);
  NUM_THREADS = min(1 << 5,cutoff_nof_runs);
  NUM_BLOCKS = (cutoff_nof_runs + NUM_THREADS - 1) / NUM_THREADS;
  find_cutoff_kernel<<<NUM_BLOCKS,NUM_THREADS,0,stream>>>(sorted_bucket_sizes,h_nof_buckets_to_compute,bucket_th,cutoff_run_length,nof_large_buckets);
  // cudaDeviceSynchronize();
  printf("cuda error cutoff %u\n",cudaGetLastError());

  unsigned h_nof_large_buckets;
  cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, stream);

  

  unsigned *max_res;
  cudaMallocAsync(&max_res, sizeof(unsigned)*2, stream);
  find_max_size<<<1,1,0,stream>>>(sorted_bucket_sizes,sorted_single_bucket_indices,c,max_res);
  // cudaDeviceSynchronize();
  printf("cuda error max %u\n",cudaGetLastError());

  unsigned h_max_res[2];
  cudaMemcpyAsync(h_max_res, max_res, sizeof(unsigned)*2, cudaMemcpyDeviceToHost, stream);
  printf("h_nof_large_buckets %u\n", h_nof_large_buckets);
  unsigned h_largest_bucket_size = h_max_res[0];
  unsigned h_nof_zero_large_buckets = h_max_res[1];
  printf("h_largest_bucket_size %u\n", h_largest_bucket_size);
  printf("h_nof_zero_large_buckets %u\n", h_nof_zero_large_buckets);

  unsigned large_buckets_to_compute = h_nof_large_buckets>h_nof_zero_large_buckets? h_nof_large_buckets-h_nof_zero_large_buckets : 0;

  cudaStream_t stream2;
  cudaStreamCreate(&stream2);
  P* large_buckets;

  if (large_buckets_to_compute>0 && bucket_th>0){
  // thrust::device_ptr<unsigned> thrust_ptr(sorted_bucket_sizes);


  // printf("sorted_bucket_sizes %u\n", sorted_bucket_sizes);
  // printf("thrust_ptr %u\n", *thrust_ptr);
  // // printf("thrust_ptr %u\n", *thrust_ptr[0]);
  // printf("thrust_ptr %u\n", thrust_ptr[0]);
  // printf("thrust_ptr %u\n", thrust_ptr[1]);
  // printf("thrust_ptr %u\n", thrust_ptr[h_nof_buckets_to_compute-1]);

  // // Perform lower_bound on the device pointer
  // thrust::device_ptr<unsigned> iter = thrust::upper_bound(thrust::seq, thrust_ptr, thrust_ptr + h_nof_buckets_to_compute, bucket_th);
  // thrust::device_ptr<unsigned> iter2 = thrust::lower_bound(thrust::seq, thrust_ptr, thrust_ptr + h_nof_buckets_to_compute, bucket_th);

  // Compute the index
  // unsigned cutoff_index = iter - thrust_ptr;
  // unsigned cutoff_index2 = iter2 - thrust_ptr;

  // printf("cutoff_index %u\n", cutoff_index);
  // printf("cutoff_index2 %u\n", cutoff_index2);
  // printf("cutoff_index %u\n", *cutoff_index);
  // printf("cutoff_index %u\n", iter);
  // printf("cutoff_index %u\n", *iter);

  // for (int i=0;;i++){
  // for (int i=0;i<3;i++){

  unsigned threads_per_bucket = 1<<(unsigned)ceil(log2((h_largest_bucket_size + bucket_th - 1) / bucket_th)); //global param
  unsigned max_bucket_size_run_length = (h_largest_bucket_size + threads_per_bucket - 1) / threads_per_bucket;
  printf("threads_per_bucket %u\n", threads_per_bucket);
  printf("max_bucket_size_run_length %u\n", max_bucket_size_run_length);




  unsigned total_large_buckets_size = large_buckets_to_compute*threads_per_bucket;
  cudaMallocAsync(&large_buckets, sizeof(P)*total_large_buckets_size, stream);
  NUM_THREADS = min(1 << 8,total_large_buckets_size);
  // NUM_THREADS = 1 << 5;
  NUM_BLOCKS = (total_large_buckets_size + NUM_THREADS - 1) / NUM_THREADS;
  accumulate_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream2>>>(large_buckets, sorted_bucket_offsets+h_nof_zero_large_buckets, sorted_bucket_sizes+h_nof_zero_large_buckets, sorted_single_bucket_indices+h_nof_zero_large_buckets, point_indices, 
  d_points, nof_buckets, large_buckets_to_compute, c+bm_bitsize, c, threads_per_bucket, max_bucket_size_run_length);                   
  // cudaDeviceSynchronize();
  printf("cuda error acc %u\n",cudaGetLastError());

//    cudaDeviceSynchronize();
// std::vector<P> h_large_buckets;
// h_large_buckets.reserve(total_large_buckets_size);
//     cudaMemcpyAsync(h_large_buckets.data(), large_buckets, sizeof(P) *total_large_buckets_size, cudaMemcpyDeviceToHost, stream);
//     std::cout<<"large buckets accumulated"<<std::endl;
//     for (unsigned i = 0; i < total_large_buckets_size; i++)
//     {
//       std::cout<<h_large_buckets[i]<<" ";
//     }
//     std::cout<<std::endl;
//reduce
  for (int s=total_large_buckets_size>>1;s>large_buckets_to_compute-1;s>>=1){
    NUM_THREADS = min(MAX_TH,s);
    NUM_BLOCKS = (s + NUM_THREADS - 1) / NUM_THREADS;
    single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream2>>>(large_buckets,large_buckets,s*2,0,0,0);
    // cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
    // cudaDeviceSynchronize();
    // printf("cuda error ones  %u\n",cudaGetLastError());
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_ones_results.data(), ones_results, sizeof(P) * nof_runs, cudaMemcpyDeviceToHost);
    // std::cout<<"one results"<<std::endl;
    // for (int i = 0; i < nof_runs; i++)
    // {
    //   std::cout<<h_ones_results[i]<<" ";
    // }
    // std::cout<<std::endl;
    }
    cudaDeviceSynchronize();
// std::vector<P> h_large_buckets;
// h_large_buckets.reserve(total_large_buckets_size);
    // cudaMemcpy(h_large_buckets.data(), large_buckets, sizeof(P) *total_large_buckets_size, cudaMemcpyDeviceToHost);
    // std::cout<<"large buckets accumulated"<<std::endl;
    // for (unsigned i = 0; i < total_large_buckets_size; i++)
    // {
    //   std::cout<<h_large_buckets[i]<<" ";
    // }
    // std::cout<<std::endl;

//distribute
  NUM_THREADS = min(MAX_TH,large_buckets_to_compute);
  NUM_BLOCKS = (large_buckets_to_compute + NUM_THREADS - 1) / NUM_THREADS;
  distribute_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream2>>>(large_buckets,buckets,sorted_single_bucket_indices+h_nof_zero_large_buckets,large_buckets_to_compute);
  // cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());


//   cudaDeviceSynchronize();
// std::vector<P> h_buckets;
// h_buckets.reserve(nof_buckets);
//     cudaMemcpy(h_buckets.data(), buckets, sizeof(P) *nof_buckets, cudaMemcpyDeviceToHost);
//     std::cout<<"buckets accumulated large"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_buckets[i]<<" ";
//     }
//     std::cout<<std::endl;
//sync stream


}
else{
  h_nof_large_buckets = 0;
}

  //launch the accumulation kernel with maximum threads
  NUM_THREADS = 1 << 8;
  // NUM_THREADS = 1 << 5;
  NUM_BLOCKS = (h_nof_buckets_to_compute-h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
  // accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, 
  //                                                        d_points, nof_buckets, nof_buckets_to_compute, c+bm_bitsize, c);                                              
  accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, sorted_bucket_offsets+h_nof_large_buckets, sorted_bucket_sizes+h_nof_large_buckets, sorted_single_bucket_indices+h_nof_large_buckets, point_indices, 
                                                          d_points, nof_buckets, h_nof_buckets_to_compute-h_nof_large_buckets, c+bm_bitsize, c);                   
   // accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, sorted_bucket_offsets, sorted_bucket_sizes, sorted_single_bucket_indices, point_indices, 
   //                                                        d_points, nof_buckets, nof_buckets_to_compute, c-1+bm_bitsize);                                              
                                                          // cudaDeviceSynchronize();
                                                          printf("cuda error acc %u\n",cudaGetLastError());
cudaStreamSynchronize(stream2);
cudaStreamDestroy(stream2);
cudaDeviceSynchronize();
// std::vector<P> h_buckets2;
// h_buckets2.reserve(nof_buckets);
//     cudaMemcpy(h_buckets2.data(), buckets, sizeof(P) *nof_buckets, cudaMemcpyDeviceToHost);
//     std::cout<<"buckets accumulated small"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_buckets2[i]<<" ";
//     }
//     std::cout<<std::endl;
// cudaDeviceSynchronize();
// cudaMemcpy(h_single.data(), single_bucket_indices, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
// cudaMemcpy(h_sizes.data(), bucket_sizes, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
// cudaMemcpy(h_offsets.data(), bucket_offsets, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
//     std::cout<<"bucket_offsets"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_offsets[i]<<" ";
//     }
//     std::cout<<std::endl;
// std::cout<<"single buckets"<<std::endl;
// for (unsigned i = 0; i < nof_buckets; i++)
// {
//   std::cout<<h_single[i]<<" ";
// }
// std::cout<<std::endl;
// std::cout<<"bucket sizes"<<std::endl;
// for (unsigned i = 0; i < nof_buckets; i++)
// {
//   std::cout<<h_sizes[i]<<" ";
// }
// std::cout<<std::endl;

// unsigned h_old_nof_buckets_to_compute = h_nof_buckets_to_compute;
// unsigned *old_single_bucket_indices = single_bucket_indices;
// unsigned *old_bucket_offsets = bucket_offsets;
// unsigned *old_bucket_sizes = bucket_sizes;
// nof_buckets_to_compute = nullptr;
// single_bucket_indices = nullptr;
// bucket_offsets = nullptr;
// bucket_sizes = nullptr;
// cudaMallocAsync(&nof_buckets_to_compute, sizeof(unsigned), stream);
// cudaMallocAsync(&single_bucket_indices, sizeof(unsigned)*h_old_nof_buckets_to_compute, stream);
// cudaMallocAsync(&bucket_offsets, sizeof(unsigned)*h_old_nof_buckets_to_compute, stream);
// cudaMallocAsync(&bucket_sizes, sizeof(unsigned)*h_old_nof_buckets_to_compute, stream);

// unsigned *sort_sizes_temp_storage{};
//   size_t sort_sizes_temp_storage_bytes;
//   cub::DeviceRadixSort::SortPairs(sort_sizes_temp_storage, sort_sizes_temp_storage_bytes, old_single_bucket_indices, single_bucket_indices,
//     old_bucket_sizes, bucket_sizes, h_old_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
//   cudaMallocAsync(&sort_indices_temp_storage, sort_sizes_temp_storage_bytes, stream);
//   cub::DeviceRadixSort::SortPairs(sort_sizes_temp_storage, sort_sizes_temp_storage_bytes, old_single_bucket_indices, single_bucket_indices,
//     old_bucket_sizes, bucket_sizes, h_old_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
//  cudaFreeAsync(sort_sizes_temp_storage, stream);

//  unsigned *sort_offsets_temp_storage{};
//  size_t sort_offsets_temp_storage_bytes;
//  cub::DeviceRadixSort::SortPairs(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, old_single_bucket_indices, single_bucket_indices,
//   old_bucket_offsets, bucket_offsets, h_old_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
//  cudaMallocAsync(&sort_indices_temp_storage, sort_offsets_temp_storage_bytes, stream);
//  cub::DeviceRadixSort::SortPairs(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, old_single_bucket_indices, single_bucket_indices,
//   old_bucket_offsets, bucket_offsets, h_old_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
// cudaFreeAsync(sort_offsets_temp_storage, stream);

// cudaDeviceSynchronize();
// cudaMemcpy(h_single.data(), single_bucket_indices, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
// cudaMemcpy(h_sizes.data(), bucket_sizes, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
// cudaMemcpy(h_offsets.data(), bucket_offsets, sizeof(unsigned) * nof_buckets, cudaMemcpyDeviceToHost);
//     std::cout<<"before encode"<<std::endl;
//     std::cout<<"bucket_offsets"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_offsets[i]<<" ";
//     }
//     std::cout<<std::endl;
// std::cout<<"single buckets"<<std::endl;
// for (unsigned i = 0; i < nof_buckets; i++)
// {
//   std::cout<<h_single[i]<<" ";
// }
// std::cout<<std::endl;
// std::cout<<"bucket sizes"<<std::endl;
// for (unsigned i = 0; i < nof_buckets; i++)
// {
//   std::cout<<h_sizes[i]<<" ";
// }
// std::cout<<std::endl;

// unsigned *encode2_temp_storage{};
// size_t encode2_temp_storage_bytes = 0;
// unsigned *useless1;
// unsigned *useless2; //try thrust::unique
// cudaMallocAsync(&useless1, sizeof(unsigned)*h_old_nof_buckets_to_compute, stream);
// cudaMallocAsync(&useless2, sizeof(unsigned)*h_old_nof_buckets_to_compute, stream);
// cub::DeviceRunLengthEncode::Encode(encode2_temp_storage, encode2_temp_storage_bytes, single_bucket_indices, useless1, useless2,
//                                       nof_buckets_to_compute, h_old_nof_buckets_to_compute, stream);
// cudaMallocAsync(&encode2_temp_storage, encode2_temp_storage_bytes, stream);
// cub::DeviceRunLengthEncode::Encode(encode2_temp_storage, encode2_temp_storage_bytes, single_bucket_indices, useless1, useless2,
//   nof_buckets_to_compute, h_old_nof_buckets_to_compute, stream);
// cudaFreeAsync(encode_temp_storage, stream);
// cudaFreeAsync(useless1, stream);
// cudaFreeAsync(useless2, stream);

// cudaFreeAsync(old_single_bucket_indices, stream);
// cudaFreeAsync(old_bucket_offsets, stream);
// cudaFreeAsync(old_bucket_sizes, stream);

// cudaMemcpyAsync(&h_nof_buckets_to_compute, nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream);

// if (h_nof_buckets_to_compute <=1) break;

// }


#else
NUM_THREADS = 1 << 8;
// NUM_THREADS = 1 << 5;
NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;
accumulate_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, points, scalars, c, nof_bms, size); 
// cudaDeviceSynchronize();
printf("cuda error 111%u\n",cudaGetLastError());
#endif
//reduce top bm
// NUM_THREADS = min(MAX_TH,(source_buckets_count>>(1+j)));
//   printf("NUM_THREADS 1 %u \n" ,NUM_THREADS);
//   NUM_BLOCKS = ((source_buckets_count>>(1+j)) + NUM_THREADS - 1) / NUM_THREADS;
//   printf("NUM_BLOCKS 1 %u \n" ,NUM_BLOCKS);
//   single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(j==0?source_buckets:temp_buckets1,j==target_bits_count-1? target_buckets: temp_buckets1,1<<(source_bits_count-j),j==target_bits_count-1? 1<<target_bits_count: 0,0,0);
//   // cudaDeviceSynchronize();
//   printf("cuda error %u\n",cudaGetLastError());

  #ifdef SSM_SUM
    //sum each bucket
    NUM_THREADS = 1 << 10;
    NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
    ssm_buckets_kernel<fake_point, fake_scalar><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, single_bucket_indices, nof_buckets, c);
   
    //sum each bucket module
    P* final_results;
    cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream);
    NUM_THREADS = 1<<c;
    NUM_BLOCKS = nof_bms;
    sum_reduction_kernel<<<NUM_BLOCKS,NUM_THREADS, 0, stream>>>(buckets, final_results);
  #endif

  P* final_results;
  if (big_triangle){
    cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = nof_bms;
    NUM_BLOCKS = 1;
    #ifdef SIGNED_DIG
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, final_results, nof_bms, c-1); //sighed digits
    #else
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, final_results, nof_bms, c); 
    #endif
    // cudaDeviceSynchronize();
    printf("cuda error %u\n",cudaGetLastError());
  }
  #ifdef ZPRIZE
  else{

  unsigned source_bits_count = c;
  unsigned source_windows_count = nof_bms;
  P *source_buckets = buckets;
  buckets = nullptr;
  P *target_buckets;
  for (unsigned i = 0;; i++) {
    const unsigned target_bits_count = (source_bits_count + 1) >> 1; //c/2=8
    const unsigned target_windows_count = source_windows_count << 1; //nof bms*2 = 32
    const unsigned target_buckets_count = target_windows_count << target_bits_count; // bms*2^c = 32*2^8
    const unsigned log_data_split =
        get_optimal_log_data_split(84, source_bits_count, target_bits_count, target_windows_count); //todo - get num of multiprossecors
    const unsigned total_buckets_count = target_buckets_count << log_data_split; //32*2^8*2^7
    cudaMallocAsync(&target_buckets, sizeof(P) * total_buckets_count, stream); //32*2^8*2^7 buckets
    NUM_THREADS = 32;
    NUM_BLOCKS = (total_buckets_count + NUM_THREADS - 1) / NUM_THREADS;
    // const unsigned block_dim = total_buckets_count < 32 ? total_buckets_count : 32;
    // const unsigned grid_dim = (total_buckets_count - 1) / block_dim.x + 1;
    split_windows_kernel_inner<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(source_bits_count, source_windows_count, source_buckets, target_buckets, total_buckets_count);
    // cudaDeviceSynchronize();
    printf("cuda error %u\n",cudaGetLastError());
    cudaFreeAsync(source_buckets, stream);

    for (unsigned j = 0; j < log_data_split; j++){
    const unsigned count = total_buckets_count >> (j + 1);
    // const unsigned block_dim = count < 32 ? count : 32;
    // const unsigned grid_dim = (count - 1) / block_dim.x + 1;
    NUM_THREADS = 32;
    NUM_BLOCKS = (count + NUM_THREADS - 1) / NUM_THREADS;
    reduce_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(target_buckets, count);
    // cudaDeviceSynchronize();
    printf("cuda error %u\n",cudaGetLastError());
    }
    if (target_bits_count == 1) {
      // P results;
      // // const unsigned result_windows_count = min(fd_q::MBC, windows_count_pass_one * bits_count_pass_one);
      const unsigned result_windows_count = bitsize;
      // if (copy_results)
      //   HANDLE_CUDA_ERROR(allocate(results, result_windows_count, pool, stream));
      // HANDLE_CUDA_ERROR(last_pass_gather(bits_count_pass_one, target_buckets, copy_results ? results : ec.results, result_windows_count, stream));
      // if (copy_results) {
      //   HANDLE_CUDA_ERROR(cudaMemcpyAsync(ec.results, results, sizeof(point_jacobian) * result_windows_count, cudaMemcpyDeviceToHost, stream));
      //   if (ec.d2h_copy_finished)
      //     HANDLE_CUDA_ERROR(cudaEventRecord(ec.d2h_copy_finished, stream));
      //   if (ec.d2h_copy_finished_callback)
      //     HANDLE_CUDA_ERROR(cudaLaunchHostFunc(stream, ec.d2h_copy_finished_callback, ec.d2h_copy_finished_callback_data));
      // }
      // if (copy_results)
      //   HANDLE_CUDA_ERROR(free(results, stream));
      // HANDLE_CUDA_ERROR(free(target_buckets, stream));
      nof_bms = bitsize;
      cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream);
      NUM_THREADS = 32;
      NUM_BLOCKS = (result_windows_count + NUM_THREADS - 1) / NUM_THREADS;
      // const dim3 block_dim = result_windows_count < 32 ? count : 32;
      // const dim3 grid_dim = (result_windows_count - 1) / block_dim.x + 1;
      last_pass_gather_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(c, target_buckets, final_results, result_windows_count);
      // cudaDeviceSynchronize();
      printf("cuda error %u\n",cudaGetLastError());
      c = 1;
      break;
    }
    source_buckets = target_buckets;
    target_buckets = nullptr;
    source_bits_count = target_bits_count;
    source_windows_count = target_windows_count;
  }
}
#else
else{
  // cudaDeviceSynchronize();
  // printf("cuda erddsdfsdfsror %u\n",cudaGetLastError());
//     cudaDeviceSynchronize();
// std::vector<P> h_buckets;
//   h_buckets.reserve(nof_buckets);
//     cudaMemcpy(h_buckets.data(), buckets, sizeof(P) * nof_buckets, cudaMemcpyDeviceToHost);
//     std::cout<<"buckets accumulated"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_buckets[i]<<" ";
//     }
//     std::cout<<std::endl;
  unsigned source_bits_count = c;
  bool odd_source_c = source_bits_count%2;
  unsigned source_windows_count = nof_bms;
  // unsigned source_window_buckets_count = 1 << source_bits_count;
  unsigned source_buckets_count = nof_buckets;
  P *source_buckets = buckets;
  buckets = nullptr;
  P *target_buckets;
  P *temp_buckets1;
  P *temp_buckets2;
  for (unsigned i = 0;; i++) {
    // if (odd_source_c){
    //   source_buckets_count = source_buckets_count<<1;
    //   P *source_buckets_padded;
    //   cudaMalloc(&source_buckets_padded, sizeof(P) * source_buckets_count);
    //   NUM_THREADS = min(MAX_TH,(source_buckets_count));
    //   NUM_BLOCKS = ((source_buckets_count) + NUM_THREADS - 1) / NUM_THREADS;
    //   pad_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(source_buckets,source_buckets_padded,1<<source_bits_count);
    //   std::vector<P> s_buckets;
    // s_buckets.reserve(source_buckets_count);
    // cudaMemcpy(s_buckets.data(), source_buckets_padded, sizeof(P) * source_buckets_count, cudaMemcpyDeviceToHost);
    // std::cout<<"source_buckets_padded"<<std::endl;
    // for (unsigned i = 0; i < source_buckets_count; i++)
    // {
    //   std::cout<<s_buckets[i]<<" ";
    // }
    // std::cout<<std::endl;
    // source_buckets=source_buckets_padded;
    // source_bits_count = source_bits_count+1;
    // }
    printf("round %u \n" ,i);
    const unsigned target_bits_count = (source_bits_count + 1) >> 1; //c/2=8
    printf("target_bits_count %u \n" ,target_bits_count);
    const unsigned target_windows_count = source_windows_count << 1; //nof bms*2 = 32
    // const unsigned target_window_buckets_count = 1 << target_bits_count; // 2^8
    const unsigned target_buckets_count = target_windows_count << target_bits_count; // bms*2^c = 32*2^8
    // const unsigned log_data_split =
    //     get_optimal_log_data_split(84, source_bits_count, target_bits_count, target_windows_count); //todo - get num of multiprossecors
    // const unsigned total_buckets_count = target_buckets_count << log_data_split; //32*2^8*2^7
    cudaMallocAsync(&target_buckets, sizeof(P) * target_buckets_count,stream); //32*2^8*2^7 buckets
    cudaMallocAsync(&temp_buckets1, sizeof(P) * source_buckets_count/2,stream); //32*2^8*2^7 buckets
    cudaMallocAsync(&temp_buckets2, sizeof(P) * source_buckets_count/2,stream); //32*2^8*2^7 buckets
    // const unsigned block_dim = total_buckets_count < 32 ? total_buckets_count : 32;
    // const unsigned grid_dim = (total_buckets_count - 1) / block_dim.x + 1;
    //input output, streams
    // reduce_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS,0,0>>>(source_buckets, target_buckets, source_windows_count>>1);
    // for (unsigned j = 0; j < target_windows_count-1; j++) //another loop
    // reduce_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS,0,0>>>(target_buckets, target_buckets, source_windows_count>>(j+2));

    // cudaStream_t stream2;
    // cudaStreamCreate(&streams[0]);
    // cudaStreamCreate(&streams[1]);
    // cudaStreamCreate(&stream2);

    // if (source_bits_count>8){
    if (source_bits_count>0){
      for(unsigned j=0;j<target_bits_count;j++){
        // cudaDeviceSynchronize();
        // std::vector<P> t1_buckets;
        // std::vector<P> t2_buckets;
        // t1_buckets.reserve(source_buckets_count);
        // t2_buckets.reserve(source_buckets_count);
        //     cudaMemcpy(t1_buckets.data(), temp_buckets1, sizeof(P) * source_buckets_count, cudaMemcpyDeviceToHost);
        //     cudaMemcpy(t2_buckets.data(), temp_buckets2, sizeof(P) * source_buckets_count, cudaMemcpyDeviceToHost);
        //     std::cout<<"0 buckets temp1"<<std::endl;
        //     for (unsigned i = 0; i < source_buckets_count; i++)
        //     {
        //       std::cout<<t1_buckets[i]<<" ";
        //     }
        //     std::cout<<std::endl;
        //     std::cout<<"0 buckets temp2"<<std::endl;
        //     for (unsigned i = 0; i < source_buckets_count; i++)
        //     {
        //       std::cout<<t2_buckets[i]<<" ";
        //     }
        //     std::cout<<std::endl;
        // if (!odd_source_c || j!=target_bits_count-1){
        // unsigned last_j = odd_source_c? target_bits_count-2 : target_bits_count-1;
        unsigned last_j = target_bits_count-1;
        NUM_THREADS = min(MAX_TH,(source_buckets_count>>(1+j)));
        // printf("NUM_THREADS 1 %u \n" ,NUM_THREADS);
        NUM_BLOCKS = ((source_buckets_count>>(1+j)) + NUM_THREADS - 1) / NUM_THREADS;
        // printf("NUM_BLOCKS 1 %u \n" ,NUM_BLOCKS);
        single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(j==0?source_buckets:temp_buckets1,j==target_bits_count-1? target_buckets: temp_buckets1,1<<(source_bits_count-j),j==target_bits_count-1? 1<<target_bits_count: 0,0,0);
        // cudaDeviceSynchronize();
        printf("cuda error %u\n",cudaGetLastError());
        // }
        // std::vector<P> t1_buckets;
        // std::vector<P> t2_buckets;
        // t1_buckets.reserve(source_buckets_count/2);
        // t2_buckets.reserve(source_buckets_count/2);
        //     cudaMemcpy(t1_buckets.data(), temp_buckets1, sizeof(P) * source_buckets_count, cudaMemcpyDeviceToHost);
        //     cudaMemcpy(t2_buckets.data(), temp_buckets2, sizeof(P) * source_buckets_count, cudaMemcpyDeviceToHost);
        //     std::cout<<"1 buckets temp1"<<std::endl;
        //     for (unsigned i = 0; i < source_buckets_count; i++)
        //     {
        //       std::cout<<t1_buckets[i]<<" ";
        //     }
        //     std::cout<<std::endl;
        //     std::cout<<"1 buckets temp2"<<std::endl;
        //     for (unsigned i = 0; i < source_buckets_count; i++)
        //     {
        //       std::cout<<t2_buckets[i]<<" ";
        //     }
        //     std::cout<<std::endl;
        //     cudaMemcpy(t1_buckets.data(), target_buckets, sizeof(P) * target_buckets_count, cudaMemcpyDeviceToHost);
        // std::cout<<"1 buckets target"<<std::endl;
        // for (unsigned i = 0; i < target_buckets_count; i++)
        // {
        //   std::cout<<t1_buckets[i]<<" ";
        // }
        // std::cout<<std::endl;
  
        // unsigned nof_threads = (source_buckets_count>>(1+j))*((odd_source_c&&j==target_bits_count-1)? 2 :1);
        unsigned nof_threads = (source_buckets_count>>(1+j));
        NUM_THREADS = min(MAX_TH,nof_threads);
        // printf("NUM_THREADS 2 %u \n" ,NUM_THREADS);
        NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
        // printf("NUM_BLOCKS 2 %u \n" ,NUM_BLOCKS);
        single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(j==0?source_buckets:temp_buckets2,j==target_bits_count-1? target_buckets: temp_buckets2,1<<(target_bits_count-j),j==target_bits_count-1? 1<<target_bits_count: 0,1,0);
        // cudaDeviceSynchronize();
        printf("cuda error %u\n",cudaGetLastError());

        // std::vector<P> t1_buckets;
        // std::vector<P> t2_buckets;
        // t1_buckets.reserve(source_buckets_count/2);
        // t2_buckets.reserve(source_buckets_count/2);
            // cudaMemcpy(t1_buckets.data(), temp_buckets1, sizeof(P) * source_buckets_count, cudaMemcpyDeviceToHost);
            // cudaMemcpy(t2_buckets.data(), temp_buckets2, sizeof(P) * source_buckets_count, cudaMemcpyDeviceToHost);
            // std::cout<<"2 buckets temp1"<<std::endl;
            // for (unsigned i = 0; i < source_buckets_count; i++)
            // {
            //   std::cout<<t1_buckets[i]<<" ";
            // }
            // std::cout<<std::endl;
            // std::cout<<"2 buckets temp2"<<std::endl;
            // for (unsigned i = 0; i < source_buckets_count; i++)
            // {
            //   std::cout<<t2_buckets[i]<<" ";
            // }
            // std::cout<<std::endl;
        
        //         std::vector<P> t1_buckets;
        // std::vector<P> t2_buckets;
        // t1_buckets.reserve(source_buckets_count/2);
        // t2_buckets.reserve(source_buckets_count/2);
        // cudaMemcpy(t1_buckets.data(), target_buckets, sizeof(P) * target_buckets_count, cudaMemcpyDeviceToHost);
        // std::cout<<"2 buckets target"<<std::endl;
        // for (unsigned i = 0; i < target_buckets_count; i++)
        // {
        //   std::cout<<t1_buckets[i]<<" ";
        // }
        // std::cout<<std::endl;

      }
    }
    else{
    // NUM_THREADS = 256;
    // NUM_BLOCKS = ((source_buckets_count>>1) + NUM_THREADS - 1) / NUM_THREADS;
    // NUM_THREADS = 1<<(source_bits_count-1);
    // printf("NUM_THREADS %u \n" ,NUM_THREADS);
    // NUM_BLOCKS = source_windows_count;
    // printf("NUM_BLOCKS %u \n" ,NUM_BLOCKS);
    // reduce_triangles_kernel<<<NUM_BLOCKS, NUM_THREADS,0,streams[0]>>>(source_buckets,temp_buckets1,target_buckets,source_bits_count,source_windows_count);
    // for(unsigned j=0;;j++){
    NUM_THREADS = 1<<(source_bits_count-1);
    printf("NUM_THREADS 1 %u \n" ,NUM_THREADS);
    NUM_BLOCKS = source_windows_count;
    printf("NUM_BLOCKS 1 %u \n" ,NUM_BLOCKS);
    general_sum_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(source_buckets,target_buckets,1<<target_bits_count,1<<target_bits_count,0);
    // cudaDeviceSynchronize();
    // printf("cuda error %u\n",cudaGetLastError());
    // cudaDeviceSynchronize();
    // std::vector<P> t_buckets;
    // t_buckets.reserve(target_buckets_count);
    //     cudaMemcpy(t_buckets.data(), target_buckets, sizeof(P) * target_buckets_count, cudaMemcpyDeviceToHost);
    //     std::cout<<"buckets target1"<<std::endl;
    //     for (unsigned i = 0; i < target_buckets_count; i++)
    //     {
    //       std::cout<<t_buckets[i]<<" ";
    //     }
    //     std::cout<<std::endl;
    // reduce_rectangles_kernel<<<NUM_BLOCKS, NUM_THREADS,0,streams[1]>>>(source_buckets,temp_buckets2,target_buckets,source_bits_count,source_windows_count);
    NUM_THREADS = 1<<(target_bits_count-1);
    printf("NUM_THREADS 2 %u \n" ,NUM_THREADS);
    NUM_BLOCKS = source_windows_count<<target_bits_count;
    printf("NUM_BLOCKS 2 %u \n" ,NUM_BLOCKS);
    general_sum_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(source_buckets,target_buckets,1,1<<target_bits_count,1);
    // cudaDeviceSynchronize();
    // printf("cuda error %u\n",cudaGetLastError());
    // }
    }
    // cudaStreamSynchronize(stream);
    // cudaStreamDestroy(streams[0]);
    // cudaStreamSynchronize(stream2);
    // cudaStreamDestroy(stream2);

    // cudaDeviceSynchronize();
    // // std::vector<P> t_buckets;
    // // t_buckets.reserve(target_buckets_count);
    //     cudaMemcpy(t_buckets.data(), target_buckets, sizeof(P) * target_buckets_count, cudaMemcpyDeviceToHost);
    //     std::cout<<"buckets target2"<<std::endl;
    //     for (unsigned i = 0; i < target_buckets_count; i++)
    //     {
    //       std::cout<<t_buckets[i]<<" ";
    //     }
    //     std::cout<<std::endl;

    // unsigned NUM_STREAMS = source_windows_count + source_windows_count*target_window_buckets_count;
    // cudaStream_t streams[NUM_STREAMS];
    // for (int k = 0; k < NUM_STREAMS; ++k) { cudaStreamCreate(&streams[k]); }

    // NUM_THREADS = 32;
    // for (unsigned j = 0; j < source_windows_count; j++){ //loop on every source bm //0-15
    //   unsigned source_offset = j*source_window_buckets_count; //0,2^16,2*2^16
    //   unsigned target_offset = j*target_window_buckets_count*2; //0,2^16,2*2^16
    //   NUM_BLOCKS = ((source_window_buckets_count>>1) + NUM_THREADS - 1) / NUM_THREADS; //2^15
    //   reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j]>>>(source_buckets+source_offset, temp_buckets1+source_offset, source_window_buckets_count>>1); //same source different target
    //   for (unsigned k = 0; k < target_bits_count-2; k++){ //0..5
    //   NUM_BLOCKS = ((source_window_buckets_count>>(k+2)) + NUM_THREADS - 1) / NUM_THREADS;//2^14..2^9
    //   reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j]>>>(temp_buckets1+source_offset, temp_buckets1+source_offset, source_window_buckets_count>>(k+2)); //stream j
    //   }
    //   NUM_BLOCKS = ((source_window_buckets_count>>target_bits_count) + NUM_THREADS - 1) / NUM_THREADS;//2^8
    //   reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j]>>>(temp_buckets1+source_offset, target_buckets+target_offset, source_window_buckets_count>>target_bits_count); //stream j
    // }

    // for (unsigned j = 0; j < source_windows_count*target_window_buckets_count; j++){ //loop on every segment of every source bm // 0..16*2^8-1
    //   unsigned source_offset = j*target_window_buckets_count;
    //   unsigned target_offset = j%target_window_buckets_count+(j/target_window_buckets_count)*target_window_buckets_count*2 + target_window_buckets_count;
    //   NUM_BLOCKS = ((target_window_buckets_count>>1) + NUM_THREADS - 1) / NUM_THREADS; //2^7
    //   reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j+source_windows_count]>>>(source_buckets+source_offset, temp_buckets2+source_offset, target_window_buckets_count>>1); //same source different target
    //   for (unsigned k = 0; k < target_bits_count-2; k++){ //0..5
    //   NUM_BLOCKS = ((target_window_buckets_count>>(k+2)) + NUM_THREADS - 1) / NUM_THREADS; //last blocks are single threaded.. //2^6..2^1
    //   reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j+source_windows_count]>>>(temp_buckets2+source_offset, temp_buckets2+source_offset, target_window_buckets_count>>(k+2));// stream j + source_windows_count
    //   }
    //   NUM_BLOCKS = 1; //last blocks are single threaded.. //
    //   reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j+source_windows_count]>>>(temp_buckets2+source_offset, target_buckets+target_offset, 1);// stream j + source_windows_count
    // }

    // for (int k = 0; k < NUM_STREAMS; ++k)
    // {
    //     cudaStreamSynchronize(streams[k]);
    //     cudaStreamDestroy(streams[k]);
    // }

    // cudaFreeAsync(source_buckets, stream);
    if (target_bits_count == 1) {
      // P results;
      // // const unsigned result_windows_count = min(fd_q::MBC, windows_count_pass_one * bits_count_pass_one);
      // const unsigned result_windows_count = bitsize;
      // if (copy_results)
      //   HANDLE_CUDA_ERROR(allocate(results, result_windows_count, pool, stream));
      // HANDLE_CUDA_ERROR(last_pass_gather(bits_count_pass_one, target_buckets, copy_results ? results : ec.results, result_windows_count, stream));
      // if (copy_results) {
      //   HANDLE_CUDA_ERROR(cudaMemcpyAsync(ec.results, results, sizeof(point_jacobian) * result_windows_count, cudaMemcpyDeviceToHost, stream));
      //   if (ec.d2h_copy_finished)
      //     HANDLE_CUDA_ERROR(cudaEventRecord(ec.d2h_copy_finished, stream));
      //   if (ec.d2h_copy_finished_callback)
      //     HANDLE_CUDA_ERROR(cudaLaunchHostFunc(stream, ec.d2h_copy_finished_callback, ec.d2h_copy_finished_callback_data));
      // }
      // if (copy_results)
      //   HANDLE_CUDA_ERROR(free(results, stream));
      // HANDLE_CUDA_ERROR(free(target_buckets, stream));
      nof_bms = bitsize;
      cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream);
      NUM_THREADS = 32;
      NUM_BLOCKS = (nof_bms + NUM_THREADS - 1) / NUM_THREADS;
      last_pass_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(target_buckets,final_results,nof_bms);
      // for (int k=0;k<bitsize;k++) {
      //   printf("k %u\n",final_results[k]);
      //   printf("k %u\n",target_buckets[k]);
      //   // final_results[k]=target_buckets[2*k+1];
      // }
      // final_results = target_buckets;
      c = 1;
      cudaFreeAsync(source_buckets,stream);
      cudaFreeAsync(target_buckets,stream);
      cudaFreeAsync(temp_buckets1,stream);
      cudaFreeAsync(temp_buckets2,stream);
      break;
    }
    cudaFreeAsync(source_buckets,stream);
    cudaFreeAsync(temp_buckets1,stream);
    cudaFreeAsync(temp_buckets2,stream);
    source_buckets = target_buckets;
    target_buckets = nullptr;
    temp_buckets1 = nullptr;
    temp_buckets2 = nullptr;
    source_bits_count = target_bits_count;
    odd_source_c = source_bits_count%2;
    source_windows_count = target_windows_count;
    // source_window_buckets_count = 1 << source_bits_count;
    source_buckets_count = target_buckets_count;
  }
}
#endif
  // cudaDeviceSynchronize();
  //   std::vector<P> h_final_results;
  //   h_final_results.reserve(nof_bms);
  //   cudaMemcpy(h_final_results.data(), final_results, sizeof(P) * nof_bms, cudaMemcpyDeviceToHost);
  //   std::cout<<"buckets summed"<<std::endl;
  //   for (unsigned i = 0; i < nof_bms; i++)
  //   {
  //     std::cout<<h_final_results[i]<<" ";
  //   }
  //   std::cout<<std::endl;


  P* d_final_result;
  if (!on_device)
    cudaMallocAsync(&d_final_result, sizeof(P), stream);

  //launch the double and add kernel, a single thread
  final_accumulation_kernel<P, S><<<1,1,0,stream>>>(final_results, ones_results, on_device ? final_result : d_final_result, 1, nof_bms, c);
  // cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
  //copy final result to host
  cudaStreamSynchronize(stream);
  if (!on_device)
    cudaMemcpyAsync(final_result, d_final_result, sizeof(P), cudaMemcpyDeviceToHost, stream);

  //free memory
  if (!on_device) {
    cudaFreeAsync(d_points, stream);
    cudaFreeAsync(d_scalars, stream);
    cudaFreeAsync(d_final_result, stream);
  }
  cudaFreeAsync(buckets, stream);
  #ifndef PHASE1_TEST
  cudaFreeAsync(bucket_indices, stream);
  cudaFreeAsync(point_indices, stream);
  cudaFreeAsync(single_bucket_indices, stream);
  cudaFreeAsync(bucket_sizes, stream);
  cudaFreeAsync(nof_buckets_to_compute, stream);
  cudaFreeAsync(bucket_offsets, stream);
  #endif
  cudaFreeAsync(sorted_bucket_sizes,stream);
  cudaFreeAsync(sorted_bucket_offsets,stream);
  cudaFreeAsync(sorted_single_bucket_indices,stream);
  cudaFreeAsync(nof_large_buckets,stream);
  cudaFreeAsync(max_res,stream);
  if (large_buckets_to_compute>0 && bucket_th>0) cudaFreeAsync(large_buckets,stream);
  cudaFreeAsync(final_results, stream);
  cudaFreeAsync(ones_results, stream);

  cudaStreamSynchronize(stream);


}

//this function computes msm using the bucket method
template <typename S, typename P, typename A>
void batched_bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned batch_size, unsigned msm_size, P* final_results, bool on_device, cudaStream_t stream){

  unsigned total_size = batch_size * msm_size;
  S *d_scalars;
  A *d_points;
  if (!on_device) {
    //copy scalars and point to gpu
    cudaMallocAsync(&d_scalars, sizeof(S) * total_size, stream);
    cudaMallocAsync(&d_points, sizeof(A) * total_size, stream);
    cudaMemcpyAsync(d_scalars, scalars, sizeof(S) * total_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_points, points, sizeof(A) * total_size, cudaMemcpyHostToDevice, stream);
  }
  else {
    d_scalars = scalars;
    d_points = points;
  }

  P *buckets;
  //compute number of bucket modules and number of buckets in each module
  unsigned nof_bms = bitsize/c;
  if (bitsize%c){
    nof_bms++;
  }
  unsigned msm_log_size = ceil(log2(msm_size));
  unsigned bm_bitsize = ceil(log2(nof_bms));
  unsigned nof_buckets = (nof_bms<<c);
  unsigned total_nof_buckets = nof_buckets*batch_size;
  cudaMallocAsync(&buckets, sizeof(P) * total_nof_buckets, stream); 

  //lanch the bucket initialization kernel with maximum threads
  unsigned NUM_THREADS = 1 << 10;
  unsigned NUM_BLOCKS = (total_nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, total_nof_buckets); 

  unsigned *bucket_indices;
  unsigned *point_indices;
  cudaMallocAsync(&bucket_indices, sizeof(unsigned) * (total_size * nof_bms + msm_size), stream);
  cudaMallocAsync(&point_indices, sizeof(unsigned) * (total_size * nof_bms + msm_size), stream);

  //split scalars into digits
  NUM_THREADS = 1 << 8;
  NUM_BLOCKS = (total_size * nof_bms + msm_size + NUM_THREADS - 1) / NUM_THREADS;
  split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(bucket_indices + msm_size, point_indices + msm_size, d_scalars, total_size, 
                                                    msm_log_size, nof_bms, bm_bitsize, c,0); //+size - leaving the first bm free for the out of place sort later

  //sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
  unsigned *sorted_bucket_indices;
  unsigned *sorted_point_indices;
  cudaMallocAsync(&sorted_bucket_indices, sizeof(unsigned) * (total_size * nof_bms), stream);
  cudaMallocAsync(&sorted_point_indices, sizeof(unsigned) * (total_size * nof_bms), stream);

  unsigned *sort_indices_temp_storage{};
  size_t sort_indices_temp_storage_bytes;
  // The second to last parameter is the default value supplied explicitly to allow passing the stream
  // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for more info
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + msm_size, sorted_bucket_indices,
                                 point_indices + msm_size, sorted_point_indices, total_size * nof_bms, 0, sizeof(unsigned)*8, stream);
  cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream);
  // The second to last parameter is the default value supplied explicitly to allow passing the stream
  // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for more info
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + msm_size, sorted_bucket_indices,
                                 point_indices + msm_size, sorted_point_indices, total_size * nof_bms, 0, sizeof(unsigned)*8, stream);
  cudaFreeAsync(sort_indices_temp_storage, stream);

  //find bucket_sizes
  unsigned *single_bucket_indices;
  unsigned *bucket_sizes;
  unsigned *total_nof_buckets_to_compute;
  cudaMallocAsync(&single_bucket_indices, sizeof(unsigned)*total_nof_buckets, stream);
  cudaMallocAsync(&bucket_sizes, sizeof(unsigned)*total_nof_buckets, stream);
  cudaMallocAsync(&total_nof_buckets_to_compute, sizeof(unsigned), stream);
  unsigned *encode_temp_storage{};
  size_t encode_temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
                                        total_nof_buckets_to_compute, nof_bms*total_size, stream);  
  cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream);
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
                                        total_nof_buckets_to_compute, nof_bms*total_size, stream);
  cudaFreeAsync(encode_temp_storage, stream);

  //get offsets - where does each new bucket begin
  unsigned* bucket_offsets;
  cudaMallocAsync(&bucket_offsets, sizeof(unsigned)*total_nof_buckets, stream);
  unsigned* offsets_temp_storage{};
  size_t offsets_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets, stream);
  cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream);
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets, stream);
  cudaFreeAsync(offsets_temp_storage, stream);

  //launch the accumulation kernel with maximum threads
  // NUM_THREADS = 1 << 8;
  // NUM_BLOCKS = (total_nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  // accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, sorted_point_indices,
  //                                                       d_points, nof_buckets, total_nof_buckets_to_compute, c+bm_bitsize,c);

  // #ifdef SSM_SUM
  //   //sum each bucket
  //   NUM_THREADS = 1 << 10;
  //   NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  //   ssm_buckets_kernel<P, S><<<NUM_BLOCKS, NUM_THREADS>>>(buckets, single_bucket_indices, nof_buckets, c);
   
  //   //sum each bucket module
  //   P* final_results;
  //   cudaMalloc(&final_results, sizeof(P) * nof_bms);
  //   NUM_THREADS = 1<<c;
  //   NUM_BLOCKS = nof_bms;
  //   sum_reduction_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(buckets, final_results);
  // #endif

  // #ifdef BIG_TRIANGLE
    P* bm_sums;
    cudaMallocAsync(&bm_sums, sizeof(P) * nof_bms * batch_size, stream);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = 1<<8;
    NUM_BLOCKS = (nof_bms*batch_size + NUM_THREADS - 1) / NUM_THREADS;
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, bm_sums, nof_bms*batch_size, c);
  // #endif

  P* d_final_results;
  if (!on_device)
    cudaMallocAsync(&d_final_results, sizeof(P)*batch_size, stream);

  //launch the double and add kernel, a single thread for each msm
  NUM_THREADS = 1<<8;
  NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS;
  final_accumulation_kernel<P, S><<<NUM_BLOCKS,NUM_THREADS, 0, stream>>>(bm_sums,bm_sums, on_device ? final_results : d_final_results, batch_size, nof_bms, c);
  
  final_accumulation_kernel<P, S><<<NUM_BLOCKS,NUM_THREADS>>>(bm_sums,bm_sums, on_device ? final_results : d_final_results, batch_size, nof_bms, c);

  //copy final result to host
  if (!on_device)
    cudaMemcpyAsync(final_results, d_final_results, sizeof(P)*batch_size, cudaMemcpyDeviceToHost, stream);

  //free memory
  if (!on_device) {
    cudaFreeAsync(d_points, stream);
    cudaFreeAsync(d_scalars, stream);
    cudaFreeAsync(d_final_results, stream);
  }
  cudaFreeAsync(buckets, stream);
  cudaFreeAsync(bucket_indices, stream);
  cudaFreeAsync(point_indices, stream);
  cudaFreeAsync(sorted_bucket_indices, stream);
  cudaFreeAsync(sorted_point_indices, stream);
  cudaFreeAsync(single_bucket_indices, stream);
  cudaFreeAsync(bucket_sizes, stream);
  cudaFreeAsync(total_nof_buckets_to_compute, stream);
  cudaFreeAsync(bucket_offsets, stream);
  cudaFreeAsync(bm_sums, stream);

  cudaStreamSynchronize(stream);
}


//this kernel converts affine points to projective points
//each thread deals with a single point
template <typename P, typename A>
__global__ void to_proj_kernel(A* affine_points, P* proj_points, unsigned N){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) proj_points[tid] = P::from_affine(affine_points[tid]);
}

//the function computes msm using ssm
template <typename S, typename P, typename A>
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result, cudaStream_t stream){ //works up to 2^8
  S *scalars;
  A *a_points;
  P *p_points;
  P *results;

  cudaMallocAsync(&scalars, sizeof(S) * size, stream);
  cudaMallocAsync(&a_points, sizeof(A) * size, stream);
  cudaMallocAsync(&p_points, sizeof(P) * size, stream);
  cudaMallocAsync(&results, sizeof(P) * size, stream);

  //copy inputs to device
  cudaMemcpyAsync(scalars, h_scalars, sizeof(S) * size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(a_points, h_points, sizeof(A) * size, cudaMemcpyHostToDevice, stream);

  //convert to projective representation and multiply each point by its scalar using single scalar multiplication
  unsigned NUM_THREADS = size;
  to_proj_kernel<<<1,NUM_THREADS, 0, stream>>>(a_points, p_points, size);
  ssm_kernel<<<1,NUM_THREADS, 0, stream>>>(scalars, p_points, results, size);

  P *final_result;
  cudaMallocAsync(&final_result, sizeof(P), stream);

  //assuming msm size is a power of 2
  //sum all the ssm results
  NUM_THREADS = size;
  sum_reduction_kernel<<<1,NUM_THREADS, 0, stream>>>(results, final_result);

  //copy result to host
  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(h_final_result, final_result, sizeof(P), cudaMemcpyDeviceToHost, stream);

  //free memory
  cudaFreeAsync(scalars, stream);
  cudaFreeAsync(a_points, stream);
  cudaFreeAsync(p_points, stream);
  cudaFreeAsync(results, stream);
  cudaFreeAsync(final_result, stream);

}

//the function computes msm on the host using the naive method
template <typename A, typename S, typename P>
void reference_msm(S* scalars, A* a_points, unsigned size){
  
  P *points = new P[size];
  // P points[size];
  for (unsigned i = 0; i < size ; i++)
  {
    points[i] = P::from_affine(a_points[i]);
  }

  P res = P::zero();
  
  for (unsigned i = 0; i < size; i++)
  {
    res = res + scalars[i]*points[i];
  }

  std::cout<<"reference results"<<std::endl;
  std::cout<<P::to_affine(res)<<std::endl;
  
}

unsigned get_optimal_c(const unsigned size) {
  if (size < 17)
    return 1;
  // return 17;
  return ceil(log2(size))-4;
}

//this function is used to compute msms of size larger than 256
template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result, bool on_device, bool big_triangle, cudaStream_t stream){
  // unsigned c = get_optimal_c(size);
  unsigned c = 16;
  // unsigned c = 8;
  unsigned bitsize = S::NBITS;
  // unsigned bitsize = 254; //get from field
  bucket_method_msm(bitsize, c, scalars, points, size, result, on_device, big_triangle, stream);
}

// this function is used to compute a batches of msms of size larger than 256
template <typename S, typename P, typename A>
void batched_large_msm(S* scalars, A* points, unsigned batch_size, unsigned msm_size, P* result, bool on_device, cudaStream_t stream){
  unsigned c = get_optimal_c(msm_size);
  // unsigned c = 6;
  // unsigned bitsize = 32;
  unsigned bitsize = 255;
  batched_bucket_method_msm(bitsize, c, scalars, points, batch_size, msm_size, result, on_device, stream);
}
#endif
