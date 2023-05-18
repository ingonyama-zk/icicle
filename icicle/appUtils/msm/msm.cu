#include <iostream>
#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "../../utils/cuda_utils.cuh"
#include "../../primitives/projective.cuh"
#include "../../primitives/field.cuh"
#include "../../curves/curve_config.cuh"
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
void bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned size, P* final_result, bool on_device) {
  
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
  unsigned nof_buckets = nof_bms<<c;
  cudaMalloc(&buckets, sizeof(P) * nof_buckets);

  // launch the bucket initialization kernel with maximum threads
  unsigned NUM_THREADS = 1 << 10;
  unsigned NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, nof_buckets);

  unsigned *bucket_indices;
  unsigned *point_indices;
  cudaMalloc(&bucket_indices, sizeof(unsigned) * size * (nof_bms+1));
  cudaMalloc(&point_indices, sizeof(unsigned) * size * (nof_bms+1));

  //split scalars into digits
  NUM_THREADS = 1 << 10;
  NUM_BLOCKS = (size * (nof_bms+1) + NUM_THREADS - 1) / NUM_THREADS;
  split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(bucket_indices + size, point_indices + size, d_scalars, size, msm_log_size, 
                                                    nof_bms, bm_bitsize, c); //+size - leaving the first bm free for the out of place sort later

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

  //launch the accumulation kernel with maximum threads
  NUM_THREADS = 1 << 8;
  NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, 
                                                         d_points, nof_buckets, 1, c+bm_bitsize);

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

  #ifdef BIG_TRIANGLE
    P* final_results;
    cudaMalloc(&final_results, sizeof(P) * nof_bms);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = nof_bms;
    NUM_BLOCKS = 1;
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, final_results, nof_bms, c);
  #endif

  P* d_final_result;
  if (!on_device)
    cudaMalloc(&d_final_result, sizeof(P));

  //launch the double and add kernel, a single thread
  final_accumulation_kernel<P, S><<<1,1>>>(final_results, on_device ? final_result : d_final_result, 1, nof_bms, c);
  
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
  // for (unsigned i = 0; i < nof_bms*batch_size; i++) {
  //   unsigned offset_out = i * msm_size;
  //   unsigned offset_in = offset_out + msm_size;
  //   cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
  //                                 bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, msm_size);
  // }
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
                                                        d_points, nof_buckets, batch_size, c+bm_bitsize);

  #ifdef SSM_SUM
    //sum each bucket
    NUM_THREADS = 1 << 10;
    NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
    ssm_buckets_kernel<P, S><<<NUM_BLOCKS, NUM_THREADS>>>(buckets, single_bucket_indices, nof_buckets, c);
   
    //sum each bucket module
    P* final_results;
    cudaMalloc(&final_results, sizeof(P) * nof_bms);
    NUM_THREADS = 1<<c;
    NUM_BLOCKS = nof_bms;
    sum_reduction_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(buckets, final_results);
  #endif

  #ifdef BIG_TRIANGLE
    P* bm_sums;
    cudaMalloc(&bm_sums, sizeof(P) * nof_bms * batch_size);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = 1<<8;
    NUM_BLOCKS = (nof_bms*batch_size + NUM_THREADS - 1) / NUM_THREADS;
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, bm_sums, nof_bms*batch_size, c);
  #endif

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
  if (size == 0)
    return 1;
  return 10;
}

//this function is used to compute msms of size larger than 256
template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result, bool on_device){
  unsigned c = get_optimal_c(size);
  // unsigned c = 6;
  // unsigned bitsize = 32;
  unsigned bitsize = 255;
  bucket_method_msm(bitsize, c, scalars, points, size, result, on_device);
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

extern "C"
int msm_cuda(projective_t *out, affine_t points[],
              scalar_t scalars[], size_t count, size_t device_id = 0)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    if (count>256){
        large_msm<scalar_t, projective_t, affine_t>(scalars, points, count, out, false);
    }
    else{
        short_msm<scalar_t, projective_t, affine_t>(scalars, points, count, out);
    }

    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int msm_batch_cuda(projective_t* out, affine_t points[],
                              scalar_t scalars[], size_t batch_size, size_t msm_size, size_t device_id = 0)
{
    // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    batched_large_msm<scalar_t, projective_t, affine_t>(scalars, points, batch_size, msm_size, out, false);

    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}

/**
 * Commit to a polynomial using the MSM.
 * Note: this function just calls the MSM, it doesn't convert between evaluation and coefficient form of scalars or points.
 * @param d_out Ouptut point to write the result to.
 * @param d_scalars Scalars for the MSM. Must be on device.
 * @param d_points Affine points for the MSM. Must be on device.
 * @param count Length of `d_scalars` and `d_points` arrays (they should have equal length).
 */
extern "C"
int commit_cuda(projective_t* d_out, scalar_t* d_scalars, affine_t* d_points, size_t count, size_t device_id = 0)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    large_msm(d_scalars, d_points, count, d_out, true);
    return 0;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}
 
 /**
  * Commit to a batch of polynomials using the MSM.
  * Note: this function just calls the MSM, it doesn't convert between evaluation and coefficient form of scalars or points.
  * @param d_out Ouptut point to write the results to.
  * @param d_scalars Scalars for the MSMs of all polynomials. Must be on device.
  * @param d_points Affine points for the MSMs. Must be on device. It is assumed that this set of bases is used for each MSM.
  * @param count Length of `d_points` array, `d_scalar` has length `count` * `batch_size`.
  * @param batch_size Size of the batch.
  */
extern "C"
int commit_batch_cuda(projective_t* d_out, scalar_t* d_scalars, affine_t* d_points, size_t count, size_t batch_size, size_t device_id = 0)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    batched_large_msm(d_scalars, d_points, batch_size, count, d_out, true);
    return 0;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}

#if defined(G2_DEFINED)
/**
 * Commit to a polynomial using the MSM in G2 group.
 * Note: this function just calls the MSM, it doesn't convert between evaluation and coefficient form of scalars or points.
 * @param d_out Ouptut G2 point to write the result to.
 * @param d_scalars Scalars for the MSM. Must be on device.
 * @param d_points G2 affine points for the MSM. Must be on device.
 * @param count Length of `d_scalars` and `d_points` arrays (they should have equal length).
 */
extern "C"
int commit_g2_cuda(g2_projective_t* d_out, scalar_t* d_scalars, g2_affine_t* d_points, size_t count, size_t device_id = 0)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    large_msm(d_scalars, d_points, count, d_out, true);
    return 0;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}
 
 /**
  * Commit to a batch of polynomials using the MSM.
  * Note: this function just calls the MSM, it doesn't convert between evaluation and coefficient form of scalars or points.
  * @param d_out Ouptut G2 point to write the results to.
  * @param d_scalars Scalars for the MSMs of all polynomials. Must be on device.
  * @param d_points G2 affine points for the MSMs. Must be on device. It is assumed that this set of bases is used for each MSM.
  * @param count Length of `d_points` array, `d_scalar` has length `count` * `batch_size`.
  * @param batch_size Size of the batch.
  */
extern "C"
int commit_batch_g2_cuda(g2_projective_t* d_out, scalar_t* d_scalars, g2_affine_t* d_points, size_t count, size_t batch_size, size_t device_id = 0)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    batched_large_msm(d_scalars, d_points, batch_size, count, d_out, true);
    return 0;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}
#endif
