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


#define BIG_TRIANGLE
// #define SSM_SUM  //WIP

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
__global__ void split_scalars_kernel(unsigned *buckets_indices, unsigned *point_indices, S *scalars, unsigned total_size, unsigned msm_log_size, unsigned nof_bms, unsigned bm_bitsize, unsigned c){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned bucket_index;
  unsigned current_index;
  unsigned msm_index = tid >> msm_log_size;
  if (tid < total_size){
    S scalar = scalars[tid];

    for (unsigned bm = 0; bm < nof_bms; bm++)
    {
      bucket_index = scalar.get_scalar_digit(bm, c);
      current_index = bm * total_size + tid;
      buckets_indices[current_index] = (msm_index<<(c+bm_bitsize)) | (bm<<c) | bucket_index;  //the bucket module number and the msm number are appended at the msbs
      point_indices[current_index] = tid; //the point index is saved for later
    }
  }
}

//this kernel adds up the points in each bucket
template <typename P, typename A>
__global__ void accumulate_buckets_kernel(P *__restrict__ buckets, unsigned *__restrict__ bucket_offsets,
               unsigned *__restrict__ bucket_sizes, unsigned *__restrict__ single_bucket_indices, unsigned *__restrict__ point_indices, A *__restrict__ points, unsigned nof_buckets, unsigned batch_size, unsigned msm_idx_shift){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned msm_index = single_bucket_indices[tid]>>msm_idx_shift;
  unsigned bucket_index = msm_index * nof_buckets + (single_bucket_indices[tid]&((1<<msm_idx_shift)-1));
  unsigned bucket_size = bucket_sizes[tid];
  if (tid>=nof_buckets*batch_size || bucket_size == 0){ //if the bucket is empty we don't need to continue
    return;
  }
  unsigned bucket_offset = bucket_offsets[tid];
  for (unsigned i = 0; i < bucket_sizes[tid]; i++)  //add the relevant points starting from the relevant offset up to the bucket size
  {
    buckets[bucket_index] = buckets[bucket_index] + points[point_indices[bucket_offset+i]];
  }
}

//this kernel sums the entire bucket module
//each thread deals with a single bucket module
template <typename P>
__global__ void big_triangle_sum_kernel(P* buckets, P* final_sums, unsigned nof_bms, unsigned c){

  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>nof_bms) return;
  P line_sum = buckets[(tid+1)*(1<<c)-1];
  final_sums[tid] = line_sum;
  for (unsigned i = (1<<c)-2; i >0; i--)
  {
    line_sum = line_sum + buckets[tid*(1<<c) + i];  //using the running sum method
    final_sums[tid] = final_sums[tid] + line_sum;
  }
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

//this function computes msm using the bucket method
template <typename S, typename P, typename A>
void bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned size, P* final_result, bool on_device, cudaStream_t stream) {
  
  S *d_scalars;
  A *d_points;
  if (!on_device) {
    //copy scalars and point to gpu
    cudaMallocAsync(&d_scalars, sizeof(S) * size, stream);
    cudaMallocAsync(&d_points, sizeof(A) * size, stream);
    cudaMemcpyAsync(d_scalars, scalars, sizeof(5) * size, cudaMemcpyHostToDevice, stream);
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
  unsigned nof_buckets = nof_bms<<c;
  cudaMallocAsync(&buckets, sizeof(P) * nof_buckets, stream);

  // launch the bucket initialization kernel with maximum threads
  unsigned NUM_THREADS = 1 << 10;
  unsigned NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, nof_buckets);

  unsigned *bucket_indices;
  unsigned *point_indices;
  cudaMallocAsync(&bucket_indices, sizeof(unsigned) * size * (nof_bms+1), stream);
  cudaMallocAsync(&point_indices, sizeof(unsigned) * size * (nof_bms+1), stream);

  //split scalars into digits
  NUM_THREADS = 1 << 10;
  NUM_BLOCKS = (size * (nof_bms+1) + NUM_THREADS - 1) / NUM_THREADS;
  split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(bucket_indices + size, point_indices + size, d_scalars, size, msm_log_size, 
                                                    nof_bms, bm_bitsize, c); //+size - leaving the first bm free for the out of place sort later

  //sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
  unsigned *sort_indices_temp_storage{};
  size_t sort_indices_temp_storage_bytes;
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + size, bucket_indices,
                                 point_indices + size, point_indices, size, 0, sizeof(unsigned) * 8, stream);
  cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream);
  for (unsigned i = 0; i < nof_bms; i++) {
    unsigned offset_out = i * size;
    unsigned offset_in = offset_out + size;
    cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage,
        sort_indices_temp_storage_bytes,
        bucket_indices + offset_in,
        bucket_indices + offset_out,
        point_indices + offset_in,
        point_indices + offset_out,
        size,
        0,
        sizeof(unsigned) * 8,
        stream
    );
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

  //get offsets - where does each new bucket begin
  unsigned* bucket_offsets;
  cudaMallocAsync(&bucket_offsets, sizeof(unsigned)*nof_buckets, stream);
  unsigned* offsets_temp_storage{};
  size_t offsets_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets, stream);
  cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream);
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets, stream);
  cudaFreeAsync(offsets_temp_storage, stream);

  //launch the accumulation kernel with maximum threads
  NUM_THREADS = 1 << 8;
  NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, 
                                                         d_points, nof_buckets, 1, c+bm_bitsize);

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

  #ifdef BIG_TRIANGLE
    P* final_results;
    cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = nof_bms;
    NUM_BLOCKS = 1;
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, final_results, nof_bms, c);
  #endif

  P* d_final_result;
  if (!on_device)
    cudaMallocAsync(&d_final_result, sizeof(P), stream);

  //launch the double and add kernel, a single thread
  final_accumulation_kernel<P, S><<<1,1,0,stream>>>(final_results, on_device ? final_result : d_final_result, 1, nof_bms, c);
  
  //copy final result to host
  cudaDeviceSynchronize();
  if (!on_device)
    cudaMemcpyAsync(final_result, d_final_result, sizeof(P), cudaMemcpyDeviceToHost, stream);

  //free memory
  if (!on_device) {
    cudaFreeAsync(d_points, stream);
    cudaFreeAsync(d_scalars, stream);
    cudaFreeAsync(d_final_result, stream);
  }
  cudaFreeAsync(buckets, stream);
  cudaFreeAsync(bucket_indices, stream);
  cudaFreeAsync(point_indices, stream);
  cudaFreeAsync(single_bucket_indices, stream);
  cudaFreeAsync(bucket_sizes, stream);
  cudaFreeAsync(nof_buckets_to_compute, stream);
  cudaFreeAsync(bucket_offsets, stream);
  cudaFreeAsync(final_results, stream);
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
                                                    msm_log_size, nof_bms, bm_bitsize, c); //+size - leaving the first bm free for the out of place sort later

  //sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
  unsigned *sorted_bucket_indices;
  unsigned *sorted_point_indices;
  cudaMallocAsync(&sorted_bucket_indices, sizeof(unsigned) * (total_size * nof_bms), stream);
  cudaMallocAsync(&sorted_point_indices, sizeof(unsigned) * (total_size * nof_bms), stream);

  unsigned *sort_indices_temp_storage{};
  size_t sort_indices_temp_storage_bytes;
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + msm_size, sorted_bucket_indices,
                                 point_indices + msm_size, sorted_point_indices, total_size * nof_bms, 0, sizeof(unsigned)*8, stream);
  cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream);
  // for (unsigned i = 0; i < nof_bms*batch_size; i++) {
  //   unsigned offset_out = i * msm_size;
  //   unsigned offset_in = offset_out + msm_size;
  //   cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
  //                                 bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, msm_size);
  // }
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
  NUM_THREADS = 1 << 8;
  NUM_BLOCKS = (total_nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, sorted_point_indices,
                                                        d_points, nof_buckets, batch_size, c+bm_bitsize);

  #ifdef SSM_SUM
    //sum each bucket
    NUM_THREADS = 1 << 10;
    NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
    ssm_buckets_kernel<P, S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, single_bucket_indices, nof_buckets, c);
   
    //sum each bucket module
    P* final_results;
    cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream);
    NUM_THREADS = 1<<c;
    NUM_BLOCKS = nof_bms;
    sum_reduction_kernel<<<NUM_BLOCKS,NUM_THREADS, 0, stream>>>(buckets, final_results);
  #endif

  #ifdef BIG_TRIANGLE
    P* bm_sums;
    cudaMallocAsync(&bm_sums, sizeof(P) * nof_bms * batch_size, stream);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = 1<<8;
    NUM_BLOCKS = (nof_bms*batch_size + NUM_THREADS - 1) / NUM_THREADS;
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, bm_sums, nof_bms*batch_size, c);
  #endif

  P* d_final_results;
  if (!on_device)
    cudaMallocAsync(&d_final_results, sizeof(P)*batch_size, stream);

  //launch the double and add kernel, a single thread for each msm
  NUM_THREADS = 1<<8;
  NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS;
  final_accumulation_kernel<P, S><<<NUM_BLOCKS,NUM_THREADS, 0, stream>>>(bm_sums, on_device ? final_results : d_final_results, batch_size, nof_bms, c);
  
  //copy final result to host
  cudaDeviceSynchronize();
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
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result, bool on_device, cudaStream_t stream){ //works up to 2^8
  
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
  cudaDeviceSynchronize();
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
  
  P points[size];
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
  return 10;
}

//this function is used to compute msms of size larger than 256
template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result, bool on_device, cudaStream_t stream){
  unsigned c = get_optimal_c(size);
  // unsigned c = 6;
  // unsigned bitsize = 32;
  unsigned bitsize = 255;
  bucket_method_msm(bitsize, c, scalars, points, size, result, on_device, stream);
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