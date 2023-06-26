#ifndef MSM
#define MSM
#pragma once
#include <stdexcept>
#include <cuda.h>
#include "../../primitives/affine.cuh"
#include <iostream>
#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "../../utils/cuda_utils.cuh"
#include "../../primitives/projective.cuh"
#include "../../primitives/field.cuh"
#include "msm.cuh"

// #define SIGNED_DIG
// #define BIG_TRIANGLE
// #define SSM_SUM  //WIP

// #define SIZE 256
// #define SHMEM_SIZE 256 * 4

// For last iteration (saves useless work)
// Use volatile to prevent caching in registers (compiler optimization)
// No __syncthreads() necessary!
template <typename P>
__device__ void warpReduce(P* shmem_ptr, int t, int first, int last) {
  for (int i=first; i>last; i>>=1){
    shmem_ptr[t] = shmem_ptr[t] + shmem_ptr[t + i];
  }
}

// __global__ void sum_reduction(int *v, int *v_r) {
// 	// Allocate shared memory
// 	// __shared__ int partial_sum[SHMEM_SIZE];
// 	int partial_sum[];

// 	// Calculate thread ID
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 	// Load elements AND do first add of reduction
// 	// Vector now 2x as long as number of threads, so scale i
// 	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

// 	// Store first partial result instead of just the elements
// 	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
// 	__syncthreads();

// 	// Start at 1/2 block stride and divide by two each iteration
// 	// Stop early (call device function instead)
// 	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
// 		// Each thread does work unless it is further than the stride
// 		if (threadIdx.x < s) {
// 			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
// 		}
// 		__syncthreads();
// 	}

// 	if (threadIdx.x < 32) {
// 		warpReduce(partial_sum, threadIdx.x);
// 	}

// 	// Let the thread 0 for this block write it's result to main memory
// 	// Result is inexed by this block
// 	if (threadIdx.x == 0) {
// 		v_r[blockIdx.x] = partial_sum[0];
// 	}
// }

template <typename P>
__global__ void reduce_triangles_kernel(P *source_buckets,P* temp_buckets, P *target_buckets, const unsigned source_c, const unsigned source_nof_bms) {
	// Allocate shared memory
	// __shared__ int partial_sum[SHMEM_SIZE];
	
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned source_nof_buckets = source_nof_bms<<source_c;
  unsigned source_nof_bm_buckets = 1<<source_c;
  const unsigned target_nof_bms = source_nof_bms<<1;
  const unsigned target_c = source_c>>1;
  // const unsigned target_nof_buckets = target_nof_bms<<target_c;
  const unsigned target_nof_bm_buckets = 1<<target_c;
  unsigned nof_threads_per_bm = source_nof_bm_buckets>>1;
  if (tid > source_nof_buckets>>1) return;
  unsigned bm_index = tid/nof_threads_per_bm;
  unsigned bm_bucket_index = tid%nof_threads_per_bm;
  unsigned bucket_index = bm_index*source_nof_bm_buckets + bm_bucket_index;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	// int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	temp_buckets[tid] = source_buckets[bucket_index] + source_buckets[bucket_index + nof_threads_per_bm];
	__syncthreads();

  if (tid ==0){ 
    for (int i=0;i<64;i++)
     {printf("%u ",temp_buckets[i]);}
     printf("\n");
    }
	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	// for (int s = blockDim.x / 2; s > 32; s >>= 1) {
	for (int s = nof_threads_per_bm/2; s > target_nof_bm_buckets/2; s >>= 1) {
		// Each thread does work unless it is further than the stride
    source_nof_bm_buckets = source_nof_bm_buckets>>1;
    nof_threads_per_bm = source_nof_bm_buckets>>1;
    bm_index = tid/nof_threads_per_bm;
    bm_bucket_index = tid%nof_threads_per_bm;
    bucket_index = bm_index*source_nof_bm_buckets + bm_bucket_index;
		if (tid < source_nof_bms*s) {
			temp_buckets[tid] = temp_buckets[bucket_index] + temp_buckets[bucket_index + s];
		}
		__syncthreads();
    if (tid ==0){ 
      for (int i=0;i<64;i++)
       {printf("%u ",temp_buckets[i]);}
       printf("\n");
      }
	}


	// if (bm_bucket_index < 32) {
	// 	warpReduce(temp_buckets, bucket_index, min(32,nof_threads_per_bm/2), target_nof_bm_buckets/2);
	// }

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (tid < source_nof_bms*target_nof_bm_buckets) {
		target_buckets[bucket_index] = temp_buckets[tid];
	}
  if (tid ==0){ 
    for (int i=0;i<64;i++)
     {printf("%u ",target_buckets[i]);}
     printf("\n");
    }
}

template <typename P>
__global__ void reduce_rectangles_kernel(P *source_buckets,P* temp_buckets, P *target_buckets, const unsigned source_c, const unsigned source_nof_bms) {
	// Allocate shared memory
	// __shared__ int partial_sum[SHMEM_SIZE];
	
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned source_nof_buckets = source_nof_bms<<source_c;
  unsigned source_nof_bm_buckets = 1<<source_c;
  const unsigned target_nof_bms = source_nof_bms<<1;
  const unsigned target_c = source_c>>1;
  // const unsigned target_nof_buckets = target_nof_bms<<target_c;
  const unsigned target_nof_bm_buckets = 1<<target_c;
  unsigned nof_threads_per_bm = source_nof_bm_buckets>>1;
  if (tid > source_nof_buckets>>1) return;
  unsigned bm_index = tid/nof_threads_per_bm;
  unsigned bm_bucket_index = tid%nof_threads_per_bm;
  unsigned bucket_index = bm_index*source_nof_bm_buckets + bm_bucket_index;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	// int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	temp_buckets[tid] = source_buckets[bucket_index] + source_buckets[bucket_index + nof_threads_per_bm];
	__syncthreads();

  if (tid ==0){ 
    for (int i=0;i<64;i++)
     {printf("%u ",temp_buckets[i]);}
     printf("\n");
    }
	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	// for (int s = blockDim.x / 2; s > 32; s >>= 1) {
	for (int s = nof_threads_per_bm/2; s > target_nof_bm_buckets/2; s >>= 1) {
		// Each thread does work unless it is further than the stride
    source_nof_bm_buckets = source_nof_bm_buckets>>1;
    nof_threads_per_bm = source_nof_bm_buckets>>1;
    bm_index = tid/nof_threads_per_bm;
    bm_bucket_index = tid%nof_threads_per_bm;
    bucket_index = bm_index*source_nof_bm_buckets + bm_bucket_index;
		if (tid < source_nof_bms*s) {
			temp_buckets[tid] = temp_buckets[bucket_index] + temp_buckets[bucket_index + s];
		}
		__syncthreads();
    if (tid ==0){ 
      for (int i=0;i<64;i++)
       {printf("%u ",temp_buckets[i]);}
       printf("\n");
      }
	}


	// if (threadIdx.x < 32) {
	// 	warpReduce(partial_sum, threadIdx.x);
	// }

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (bm_bucket_index < target_nof_bm_buckets) {
		target_buckets[bucket_index] = temp_buckets[bucket_index];
	}
  if (tid ==0){ 
    for (int i=0;i<64;i++)
     {printf("%u ",target_buckets[i]);}
     printf("\n");
    }
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
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned bucket_index;
  unsigned bucket_index2;
  unsigned current_index;
  unsigned msm_index = tid >> msm_log_size;
  unsigned borrow = 0;
  if (tid < total_size){
    S scalar = scalars[tid];
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
      buckets_indices[current_index] = (msm_index<<(c+bm_bitsize)) | (bm<<c) | bucket_index;  //the bucket module number and the msm number are appended at the msbs
      // buckets_indices[current_index] = (msm_index<<(c-1+bm_bitsize)) | (bm<<(c-1)) | bucket_index;  //the bucket module number and the msm number are appended at the msbs
      #ifdef SIGNED_DIG
      point_indices[current_index] = sign | tid; //the point index is saved for later
      #else
      point_indices[current_index] = tid; //the point index is saved for later
      #endif
    }
  }
}

//this kernel adds up the points in each bucket
// __global__ void accumulate_buckets_kernel(P *__restrict__ buckets, unsigned *__restrict__ bucket_offsets,
  //  unsigned *__restrict__ bucket_sizes, unsigned *__restrict__ single_bucket_indices, unsigned *__restrict__ point_indices, A *__restrict__ points, unsigned nof_buckets, unsigned batch_size, unsigned msm_idx_shift){
template <typename P, typename A>
__global__ void accumulate_buckets_kernel(P *__restrict__ buckets, const unsigned *__restrict__ bucket_offsets, const unsigned *__restrict__ bucket_sizes, const unsigned *__restrict__ single_bucket_indices, const unsigned *__restrict__ point_indices, A *__restrict__ points, const unsigned nof_buckets, const unsigned *nof_buckets_to_compute, const unsigned msm_idx_shift, const unsigned c){
  
  constexpr unsigned sign_mask = 0x80000000;
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  // if (tid>=*nof_buckets_to_compute || tid<11){ 
  if (tid>=*nof_buckets_to_compute){ 
    return;
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
  // if (tid<10) printf("tid %u size %u\n", tid, bucket_sizes[tid]);
  // if (tid==0) return;
  // if ((bucket_index>>20)==13) return;
  // if (bucket_sizes[tid]==16777216) printf("tid %u size %u bucket %u offset %u\n", tid, bucket_sizes[tid], bucket_index, bucket_offset);
  // const unsigned *indexes = point_indices + bucket_offset;
  P bucket = P::zero(); //todo: get rid of init buckets? no.. because what about buckets with no points
  // unsigned point_ind;
  for (unsigned i = 0; i < bucket_sizes[tid]; i++)  //add the relevant points starting from the relevant offset up to the bucket size
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
    bucket = bucket + point;
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

//this kernel computes the final result using the double and add algorithm
//it is done by a single thread
template <typename P, typename S>
__global__ void final_accumulation_kernel(P* final_sums, P* final_results, unsigned nof_msms, unsigned nof_bms, unsigned c){
  
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
  final_results[tid] = final_result + final_sums[tid*nof_bms];

}

template <typename P>
void test_reduce_triangle(P* h_buckets){
  for (int i=0; i<64; i++) std::cout<<h_buckets[i]<<" ";
  std::cout<<std::endl;
  P*buckets;
  P*temp;
  P*target;
  unsigned count = 64;
  cudaMalloc(&buckets, sizeof(P) * count);
  cudaMemcpy(buckets, h_buckets, sizeof(P) * count, cudaMemcpyHostToDevice);
  cudaMalloc(&temp, sizeof(P) * count);
  cudaMalloc(&target, sizeof(P) * count);
  reduce_triangles_kernel<<<2,32>>>(buckets,temp,target,4,4);
  cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
}


//this function computes msm using the bucket method
template <typename S, typename P, typename A>
void bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned size, P* final_result, bool on_device, bool big_triangle) {
  
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
    cudaMalloc(&d_scalars, sizeof(S) * size);
    cudaMalloc(&d_points, sizeof(A) * size);
    cudaMemcpy(d_scalars, scalars, sizeof(S) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, points, sizeof(A) * size, cudaMemcpyHostToDevice);
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
  // std::cout << "top_bm_nof_missing_bits" << top_bm_nof_missing_bits <<std::endl;
  // unsigned nof_buckets = nof_bms<<c;
  #ifdef SIGNED_DIG
  unsigned nof_buckets = nof_bms*((1<<(c-1))+1); //signed digits
  #else
  unsigned nof_buckets = nof_bms<<c;
  #endif
  cudaMalloc(&buckets, sizeof(P) * nof_buckets);

  // launch the bucket initialization kernel with maximum threads
  unsigned NUM_THREADS = 1 << 10;
  unsigned NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, nof_buckets);
  cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());

  unsigned *bucket_indices;
  unsigned *point_indices;
  cudaMalloc(&bucket_indices, sizeof(unsigned) * size * (nof_bms+1));
  cudaMalloc(&point_indices, sizeof(unsigned) * size * (nof_bms+1));

  //split scalars into digits
  NUM_THREADS = 1 << 10;
  NUM_BLOCKS = (size * (nof_bms+1) + NUM_THREADS - 1) / NUM_THREADS;
  split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(bucket_indices + size, point_indices + size, d_scalars, size, msm_log_size, 
                                                    nof_bms, bm_bitsize, c, top_bm_nof_missing_bits); //+size - leaving the first bm free for the out of place sort later
                                                    cudaDeviceSynchronize();
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
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + size, bucket_indices,
                                 point_indices + size, point_indices, size);

  cudaMalloc(&sort_indices_temp_storage, sort_indices_temp_storage_bytes);
  for (unsigned i = 0; i < nof_bms; i++) {
    unsigned offset_out = i * size;
    unsigned offset_in = offset_out + size;
    cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
                                  bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, size);
  }
  cudaFree(sort_indices_temp_storage);

  //find bucket_sizes
  unsigned *single_bucket_indices;
  unsigned *bucket_sizes;
  unsigned *nof_buckets_to_compute;
  cudaMalloc(&single_bucket_indices, sizeof(unsigned)*nof_buckets);
  cudaMalloc(&bucket_sizes, sizeof(unsigned)*nof_buckets);
  cudaMalloc(&nof_buckets_to_compute, sizeof(unsigned));
  unsigned *encode_temp_storage{};
  size_t encode_temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, nof_bms*size);
  cudaMalloc(&encode_temp_storage, encode_temp_storage_bytes);
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, nof_bms*size);
  cudaFree(encode_temp_storage);

  //get offsets - where does each new bucket begin
  unsigned* bucket_offsets;
  cudaMalloc(&bucket_offsets, sizeof(unsigned)*nof_buckets);
  unsigned* offsets_temp_storage{};
  size_t offsets_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets);
  cudaMalloc(&offsets_temp_storage, offsets_temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets);
  cudaFree(offsets_temp_storage);

  //sort by bucket sizes
  unsigned* sorted_bucket_sizes;
  unsigned* sorted_bucket_offsets;
  unsigned* sorted_single_bucket_indices;
  cudaMalloc(&sorted_bucket_sizes, sizeof(unsigned)*nof_buckets);
  cudaMalloc(&sorted_bucket_offsets, sizeof(unsigned)*nof_buckets);
  cudaMalloc(&sorted_single_bucket_indices, sizeof(unsigned)*nof_buckets);
  unsigned* sort_offsets_temp_storage{};
  size_t sort_offsets_temp_storage_bytes = 0;
  unsigned* sort_single_temp_storage{};
  size_t sort_single_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, bucket_offsets, sorted_bucket_offsets, nof_buckets);
  cub::DeviceRadixSort::SortPairsDescending(sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, single_bucket_indices, sorted_single_bucket_indices, nof_buckets);
  cudaMalloc(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes);
  cudaMalloc(&sort_single_temp_storage, sort_single_temp_storage_bytes);
  cub::DeviceRadixSort::SortPairsDescending(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, bucket_offsets, sorted_bucket_offsets, nof_buckets);
  cub::DeviceRadixSort::SortPairsDescending(sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes,
    sorted_bucket_sizes, single_bucket_indices, sorted_single_bucket_indices, nof_buckets);
  cudaFree(sort_offsets_temp_storage);
  cudaFree(sort_single_temp_storage);
  

  //launch the accumulation kernel with maximum threads
  NUM_THREADS = 1 << 8;
  // NUM_THREADS = 1 << 5;
  NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  // accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, 
                                                        //  d_points, nof_buckets, nof_buckets_to_compute, c+bm_bitsize);                                              
  accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, sorted_bucket_offsets, sorted_bucket_sizes, sorted_single_bucket_indices, point_indices, 
                                                         d_points, nof_buckets, nof_buckets_to_compute, c+bm_bitsize, c);                   
  // accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, sorted_bucket_offsets, sorted_bucket_sizes, sorted_single_bucket_indices, point_indices, 
  //                                                        d_points, nof_buckets, nof_buckets_to_compute, c-1+bm_bitsize);                                              
                                                         cudaDeviceSynchronize();
                                                         printf("cuda error %u\n",cudaGetLastError());

//   cudaDeviceSynchronize();
// std::vector<P> h_buckets;
//   h_buckets.reserve(nof_buckets);
//     cudaMemcpy(h_buckets.data(), buckets, sizeof(P) * nof_buckets, cudaMemcpyDeviceToHost);
//     std::cout<<"buckets accumulated"<<std::endl;
//     for (unsigned i = 0; i < nof_buckets; i++)
//     {
//       std::cout<<h_buckets[i]<<" ";
//     }
//     std::cout<<std::endl;
  #ifdef SSM_SUM
    //sum each bucket
    NUM_THREADS = 1 << 10;
    NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
    ssm_buckets_kernel<fake_point, fake_scalar><<<NUM_BLOCKS, NUM_THREADS>>>(buckets, single_bucket_indices, nof_buckets, c);
   
    //sum each bucket module
    P* final_results;
    cudaMalloc(&final_results, sizeof(P) * nof_bms);
    NUM_THREADS = 1<<c;
    NUM_BLOCKS = nof_bms;
    sum_reduction_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(buckets, final_results);
  #endif

  P* final_results;
  if (big_triangle){
    cudaMalloc(&final_results, sizeof(P) * nof_bms);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = nof_bms;
    NUM_BLOCKS = 1;
    #ifdef SIGNED_DIG
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, final_results, nof_bms, c-1); //sighed digits
    #else
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, final_results, nof_bms, c); 
    #endif
    cudaDeviceSynchronize();
    printf("cuda error %u\n",cudaGetLastError());
  }
//   else{

//   unsigned source_bits_count = c;
//   unsigned source_windows_count = nof_bms;
//   P *source_buckets = buckets;
//   buckets = nullptr;
//   P *target_buckets;
//   for (unsigned i = 0;; i++) {
//     const unsigned target_bits_count = (source_bits_count + 1) >> 1; //c/2=8
//     const unsigned target_windows_count = source_windows_count << 1; //nof bms*2 = 32
//     const unsigned target_buckets_count = target_windows_count << target_bits_count; // bms*2^c = 32*2^8
//     const unsigned log_data_split =
//         get_optimal_log_data_split(84, source_bits_count, target_bits_count, target_windows_count); //todo - get num of multiprossecors
//     const unsigned total_buckets_count = target_buckets_count << log_data_split; //32*2^8*2^7
//     cudaMalloc(&target_buckets, sizeof(P) * total_buckets_count); //32*2^8*2^7 buckets
//     NUM_THREADS = 32;
//     NUM_BLOCKS = (total_buckets_count + NUM_THREADS - 1) / NUM_THREADS;
//     // const unsigned block_dim = total_buckets_count < 32 ? total_buckets_count : 32;
//     // const unsigned grid_dim = (total_buckets_count - 1) / block_dim.x + 1;
//     split_windows_kernel_inner<<<NUM_BLOCKS, NUM_THREADS>>>(source_bits_count, source_windows_count, source_buckets, target_buckets, total_buckets_count);
//     cudaFree(source_buckets);

//     for (unsigned j = 0; j < log_data_split; j++){
//     const unsigned count = total_buckets_count >> (j + 1);
//     // const unsigned block_dim = count < 32 ? count : 32;
//     // const unsigned grid_dim = (count - 1) / block_dim.x + 1;
//     NUM_THREADS = 32;
//     NUM_BLOCKS = (count + NUM_THREADS - 1) / NUM_THREADS;
//     reduce_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(target_buckets, count);
//     }
//     if (target_bits_count == 1) {
//       // P results;
//       // // const unsigned result_windows_count = min(fd_q::MBC, windows_count_pass_one * bits_count_pass_one);
//       const unsigned result_windows_count = bitsize;
//       // if (copy_results)
//       //   HANDLE_CUDA_ERROR(allocate(results, result_windows_count, pool, stream));
//       // HANDLE_CUDA_ERROR(last_pass_gather(bits_count_pass_one, target_buckets, copy_results ? results : ec.results, result_windows_count, stream));
//       // if (copy_results) {
//       //   HANDLE_CUDA_ERROR(cudaMemcpyAsync(ec.results, results, sizeof(point_jacobian) * result_windows_count, cudaMemcpyDeviceToHost, stream));
//       //   if (ec.d2h_copy_finished)
//       //     HANDLE_CUDA_ERROR(cudaEventRecord(ec.d2h_copy_finished, stream));
//       //   if (ec.d2h_copy_finished_callback)
//       //     HANDLE_CUDA_ERROR(cudaLaunchHostFunc(stream, ec.d2h_copy_finished_callback, ec.d2h_copy_finished_callback_data));
//       // }
//       // if (copy_results)
//       //   HANDLE_CUDA_ERROR(free(results, stream));
//       // HANDLE_CUDA_ERROR(free(target_buckets, stream));
//       nof_bms = bitsize;
//       cudaMalloc(&final_results, sizeof(P) * nof_bms);
//       NUM_THREADS = 32;
//       NUM_BLOCKS = (result_windows_count + NUM_THREADS - 1) / NUM_THREADS;
//       // const dim3 block_dim = result_windows_count < 32 ? count : 32;
//       // const dim3 grid_dim = (result_windows_count - 1) / block_dim.x + 1;
//       last_pass_gather_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(c, target_buckets, final_results, result_windows_count);
//       c = 1;
//       break;
//     }
//     source_buckets = target_buckets;
//     target_buckets = nullptr;
//     source_bits_count = target_bits_count;
//     source_windows_count = target_windows_count;
//   }
// }
else{
  unsigned source_bits_count = c;
  unsigned source_windows_count = nof_bms;
  unsigned source_window_buckets_count = 1 << source_bits_count;
  unsigned source_buckets_count = nof_buckets;
  P *source_buckets = buckets;
  buckets = nullptr;
  P *target_buckets;
  P *temp_buckets1;
  P *temp_buckets2;
  cudaMalloc(&temp_buckets1, sizeof(P) * source_buckets_count); //32*2^8*2^7 buckets
  cudaMalloc(&temp_buckets2, sizeof(P) * source_buckets_count); //32*2^8*2^7 buckets
  for (unsigned i = 0;; i++) {
    const unsigned target_bits_count = (source_bits_count + 1) >> 1; //c/2=8
    const unsigned target_windows_count = source_windows_count << 1; //nof bms*2 = 32
    const unsigned target_window_buckets_count = 1 << target_bits_count; // 2^8
    const unsigned target_buckets_count = target_windows_count << target_bits_count; // bms*2^c = 32*2^8
    // const unsigned log_data_split =
    //     get_optimal_log_data_split(84, source_bits_count, target_bits_count, target_windows_count); //todo - get num of multiprossecors
    // const unsigned total_buckets_count = target_buckets_count << log_data_split; //32*2^8*2^7
    cudaMalloc(&target_buckets, sizeof(P) * target_buckets_count); //32*2^8*2^7 buckets
   
    // const unsigned block_dim = total_buckets_count < 32 ? total_buckets_count : 32;
    // const unsigned grid_dim = (total_buckets_count - 1) / block_dim.x + 1;
    //input output, streams
    // reduce_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS,0,0>>>(source_buckets, target_buckets, source_windows_count>>1);
    // for (unsigned j = 0; j < target_windows_count-1; j++) //another loop
    // reduce_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS,0,0>>>(target_buckets, target_buckets, source_windows_count>>(j+2));
    unsigned NUM_STREAMS = source_windows_count + source_windows_count*target_window_buckets_count;
    cudaStream_t streams[NUM_STREAMS];
    for (int k = 0; k < NUM_STREAMS; ++k) { cudaStreamCreate(&streams[k]); }

    NUM_THREADS = 32;
    for (unsigned j = 0; j < source_windows_count; j++){ //loop on every source bm //0-15
      unsigned source_offset = j*source_window_buckets_count; //0,2^16,2*2^16
      unsigned target_offset = j*target_window_buckets_count*2; //0,2^16,2*2^16
      NUM_BLOCKS = ((source_window_buckets_count>>1) + NUM_THREADS - 1) / NUM_THREADS; //2^15
      reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j]>>>(source_buckets+source_offset, temp_buckets1+source_offset, source_window_buckets_count>>1); //same source different target
      for (unsigned k = 0; k < target_bits_count-2; k++){ //0..5
      NUM_BLOCKS = ((source_window_buckets_count>>(k+2)) + NUM_THREADS - 1) / NUM_THREADS;//2^14..2^9
      reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j]>>>(temp_buckets1+source_offset, temp_buckets1+source_offset, source_window_buckets_count>>(k+2)); //stream j
      }
      NUM_BLOCKS = ((source_window_buckets_count>>target_bits_count) + NUM_THREADS - 1) / NUM_THREADS;//2^8
      reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j]>>>(temp_buckets1+source_offset, target_buckets+target_offset, source_window_buckets_count>>target_bits_count); //stream j
    }

    for (unsigned j = 0; j < source_windows_count*target_window_buckets_count; j++){ //loop on every segment of every source bm // 0..16*2^8-1
      unsigned source_offset = j*target_window_buckets_count;
      unsigned target_offset = j%target_window_buckets_count+(j/target_window_buckets_count)*target_window_buckets_count*2 + target_window_buckets_count;
      NUM_BLOCKS = ((target_window_buckets_count>>1) + NUM_THREADS - 1) / NUM_THREADS; //2^7
      reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j+source_windows_count]>>>(source_buckets+source_offset, temp_buckets2+source_offset, target_window_buckets_count>>1); //same source different target
      for (unsigned k = 0; k < target_bits_count-2; k++){ //0..5
      NUM_BLOCKS = ((target_window_buckets_count>>(k+2)) + NUM_THREADS - 1) / NUM_THREADS; //last blocks are single threaded.. //2^6..2^1
      reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j+source_windows_count]>>>(temp_buckets2+source_offset, temp_buckets2+source_offset, target_window_buckets_count>>(k+2));// stream j + source_windows_count
      }
      NUM_BLOCKS = 1; //last blocks are single threaded.. //
      reduce_buckets_kernel2<<<NUM_BLOCKS, NUM_THREADS,0,streams[j+source_windows_count]>>>(temp_buckets2+source_offset, target_buckets+target_offset, 1);// stream j + source_windows_count
    }

    for (int k = 0; k < NUM_STREAMS; ++k)
    {
        cudaStreamSynchronize(streams[k]);
        cudaStreamDestroy(streams[k]);
    }

    cudaFree(source_buckets);
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
      cudaMalloc(&final_results, sizeof(P) * nof_bms);
      c = 1;
      break;
    }
    source_buckets = target_buckets;
    target_buckets = nullptr;
    source_bits_count = target_bits_count;
    source_windows_count = target_windows_count;
    source_window_buckets_count = 1 << source_bits_count;
    source_buckets_count = target_buckets_count;
  }
}

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
    cudaMalloc(&d_final_result, sizeof(P));

  //launch the double and add kernel, a single thread
  final_accumulation_kernel<P, S><<<1,1>>>(final_results, on_device ? final_result : d_final_result, 1, nof_bms, c);
  cudaDeviceSynchronize();
  printf("cuda error %u\n",cudaGetLastError());
  //copy final result to host
  cudaDeviceSynchronize();
  if (!on_device)
    cudaMemcpy(final_result, d_final_result, sizeof(P), cudaMemcpyDeviceToHost);

  //free memory
  if (!on_device) {
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_final_result);
  }
  cudaFree(buckets);
  cudaFree(bucket_indices);
  cudaFree(point_indices);
  cudaFree(single_bucket_indices);
  cudaFree(bucket_sizes);
  cudaFree(nof_buckets_to_compute);
  cudaFree(bucket_offsets);
  cudaFree(sorted_bucket_sizes);
  cudaFree(sorted_bucket_offsets);
  cudaFree(sorted_single_bucket_indices);
  cudaFree(final_results);
}

//this function computes msm using the bucket method
template <typename S, typename P, typename A>
void batched_bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned batch_size, unsigned msm_size, P* final_results, bool on_device){

  unsigned total_size = batch_size * msm_size;
  S *d_scalars;
  A *d_points;
  if (!on_device) {
    //copy scalars and point to gpu
    cudaMalloc(&d_scalars, sizeof(S) * total_size);
    cudaMalloc(&d_points, sizeof(A) * total_size);
    cudaMemcpy(d_scalars, scalars, sizeof(S) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, points, sizeof(A) * total_size, cudaMemcpyHostToDevice);
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
  cudaMalloc(&buckets, sizeof(P) * total_nof_buckets); 

  //lanch the bucket initialization kernel with maximum threads
  unsigned NUM_THREADS = 1 << 10;
  unsigned NUM_BLOCKS = (total_nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, total_nof_buckets); 

  unsigned *bucket_indices;
  unsigned *point_indices;
  cudaMalloc(&bucket_indices, sizeof(unsigned) * (total_size * nof_bms + msm_size));
  cudaMalloc(&point_indices, sizeof(unsigned) * (total_size * nof_bms + msm_size));

  //split scalars into digits
  NUM_THREADS = 1 << 8;
  NUM_BLOCKS = (total_size * nof_bms + msm_size + NUM_THREADS - 1) / NUM_THREADS;
  split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(bucket_indices + msm_size, point_indices + msm_size, d_scalars, total_size, 
                                                    msm_log_size, nof_bms, bm_bitsize, c); //+size - leaving the first bm free for the out of place sort later

  //sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
  unsigned *sorted_bucket_indices;
  unsigned *sorted_point_indices;
  cudaMalloc(&sorted_bucket_indices, sizeof(unsigned) * (total_size * nof_bms));
  cudaMalloc(&sorted_point_indices, sizeof(unsigned) * (total_size * nof_bms));

  unsigned *sort_indices_temp_storage{};
  size_t sort_indices_temp_storage_bytes;
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + msm_size, sorted_bucket_indices,
                                 point_indices + msm_size, sorted_point_indices, total_size * nof_bms);
  cudaMalloc(&sort_indices_temp_storage, sort_indices_temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + msm_size, sorted_bucket_indices,
                                 point_indices + msm_size, sorted_point_indices, total_size * nof_bms);
  cudaFree(sort_indices_temp_storage);

  //find bucket_sizes
  unsigned *single_bucket_indices;
  unsigned *bucket_sizes;
  unsigned *total_nof_buckets_to_compute;
  cudaMalloc(&single_bucket_indices, sizeof(unsigned)*total_nof_buckets);
  cudaMalloc(&bucket_sizes, sizeof(unsigned)*total_nof_buckets);
  cudaMalloc(&total_nof_buckets_to_compute, sizeof(unsigned));
  unsigned *encode_temp_storage{};
  size_t encode_temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
                                        total_nof_buckets_to_compute, nof_bms*total_size);
  cudaMalloc(&encode_temp_storage, encode_temp_storage_bytes);
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
                                        total_nof_buckets_to_compute, nof_bms*total_size);
  cudaFree(encode_temp_storage);

  //get offsets - where does each new bucket begin
  unsigned* bucket_offsets;
  cudaMalloc(&bucket_offsets, sizeof(unsigned)*total_nof_buckets);
  unsigned* offsets_temp_storage{};
  size_t offsets_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets);
  cudaMalloc(&offsets_temp_storage, offsets_temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets);
  cudaFree(offsets_temp_storage);

  //launch the accumulation kernel with maximum threads
  NUM_THREADS = 1 << 8;
  NUM_BLOCKS = (total_nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, sorted_point_indices,
                                                        d_points, nof_buckets, total_nof_buckets_to_compute, c+bm_bitsize);

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
    cudaMalloc(&bm_sums, sizeof(P) * nof_bms * batch_size);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = 1<<8;
    NUM_BLOCKS = (nof_bms*batch_size + NUM_THREADS - 1) / NUM_THREADS;
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, bm_sums, nof_bms*batch_size, c);
  // #endif

  P* d_final_results;
  if (!on_device)
    cudaMalloc(&d_final_results, sizeof(P)*batch_size);

  //launch the double and add kernel, a single thread for each msm
  NUM_THREADS = 1<<8;
  NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS;
  final_accumulation_kernel<P, S><<<NUM_BLOCKS,NUM_THREADS>>>(bm_sums, on_device ? final_results : d_final_results, batch_size, nof_bms, c);

  //copy final result to host
  cudaDeviceSynchronize();
  if (!on_device)
    cudaMemcpy(final_results, d_final_results, sizeof(P)*batch_size, cudaMemcpyDeviceToHost);

  //free memory
  if (!on_device) {
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_final_results);
  }
  cudaFree(buckets);
  cudaFree(bucket_indices);
  cudaFree(point_indices);
  cudaFree(sorted_bucket_indices);
  cudaFree(sorted_point_indices);
  cudaFree(single_bucket_indices);
  cudaFree(bucket_sizes);
  cudaFree(total_nof_buckets_to_compute);
  cudaFree(bucket_offsets);
  cudaFree(bm_sums);

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
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result){ //works up to 2^8
  S *scalars;
  A *a_points;
  P *p_points;
  P *results;

  cudaMalloc(&scalars, sizeof(S) * size);
  cudaMalloc(&a_points, sizeof(A) * size);
  cudaMalloc(&p_points, sizeof(P) * size);
  cudaMalloc(&results, sizeof(P) * size);

  //copy inputs to device
  cudaMemcpy(scalars, h_scalars, sizeof(S) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(a_points, h_points, sizeof(A) * size, cudaMemcpyHostToDevice);

  //convert to projective representation and multiply each point by its scalar using single scalar multiplication
  unsigned NUM_THREADS = size;
  to_proj_kernel<<<1,NUM_THREADS>>>(a_points, p_points, size);
  ssm_kernel<<<1,NUM_THREADS>>>(scalars, p_points, results, size);

  P *final_result;
  cudaMalloc(&final_result, sizeof(P));

  //assuming msm size is a power of 2
  //sum all the ssm results
  NUM_THREADS = size;
  sum_reduction_kernel<<<1,NUM_THREADS>>>(results, final_result);

  //copy result to host
  cudaDeviceSynchronize();
  cudaMemcpy(h_final_result, final_result, sizeof(P), cudaMemcpyDeviceToHost);

  //free memory
  cudaFree(scalars);
  cudaFree(a_points);
  cudaFree(p_points);
  cudaFree(results);
  cudaFree(final_result);

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
void large_msm(S* scalars, A* points, unsigned size, P* result, bool on_device, bool big_triangle){
  // unsigned c = get_optimal_c(size);
  unsigned c = 16;
  // unsigned bitsize = 32;
  unsigned bitsize = 253; //get from field
  bucket_method_msm(bitsize, c, scalars, points, size, result, on_device, big_triangle);
}

// this function is used to compute a batches of msms of size larger than 256
template <typename S, typename P, typename A>
void batched_large_msm(S* scalars, A* points, unsigned batch_size, unsigned msm_size, P* result, bool on_device){
  unsigned c = get_optimal_c(msm_size);
  // unsigned c = 6;
  // unsigned bitsize = 32;
  unsigned bitsize = 255;
  batched_bucket_method_msm(bitsize, c, scalars, points, batch_size, msm_size, result, on_device);
}
#endif
