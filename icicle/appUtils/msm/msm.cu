#include "msm.cuh"

#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "curves/curve_config.cuh"
#include "primitives/affine.cuh"
#include "primitives/field.cuh"
#include "primitives/projective.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/error_handler.cuh"
#include "utils/mont.cuh"
#include "utils/utils.h"

namespace msm {

  namespace {

#define MAX_TH 256

    // #define SIGNED_DIG //WIP
    // #define BIG_TRIANGLE
    // #define SSM_SUM  //WIP

    template <typename S>
    int get_optimal_c(int bitsize)
    {
      return max((int)ceil(log2(bitsize)) - 4, 1);
    }

    template <typename P>
    __global__ void single_stage_multi_reduction_kernel(
      P* v,
      P* v_r,
      unsigned block_size,
      unsigned write_stride,
      unsigned write_phase,
      unsigned padding,
      unsigned num_of_threads)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= num_of_threads) { return; }

      int jump = block_size / 2;
      int tid_p = padding ? (tid / (2 * padding)) * padding + tid % padding : tid;
      int block_id = tid_p / jump;
      int block_tid = tid_p % jump;
      unsigned read_ind = block_size * block_id + block_tid;
      unsigned write_ind = tid;
      unsigned v_r_key = write_stride
                           ? ((write_ind / write_stride) * 2 + write_phase) * write_stride + write_ind % write_stride
                           : write_ind;
      P v_r_value = padding ? (tid % (2 * padding) < padding) ? v[read_ind] + v[read_ind + jump] : P::zero()
                            : v[read_ind] + v[read_ind + jump];

      v_r[v_r_key] = v_r_value;
    }

    // this kernel performs single scalar multiplication
    // each thread multilies a single scalar and point
    template <typename P, typename S>
    __global__ void ssm_kernel(S* scalars, P* points, P* results, unsigned N)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid < N) results[tid] = scalars[tid] * points[tid];
    }

    // this kernel sums all the elements in a given vector using multiple threads
    template <typename P>
    __global__ void sum_reduction_kernel(P* v, P* v_r)
    {
      unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

      // Start at 1/2 block stride and divide by two each iteration
      for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) { v[tid] = v[tid] + v[tid + s]; }
        __syncthreads();
      }

      // Let the thread 0 for this block write the final result
      if (threadIdx.x == 0) { v_r[blockIdx.x] = v[tid]; }
    }

    // this kernel initializes the buckets with zero points
    // each thread initializes a different bucket
    template <typename P>
    __global__ void initialize_buckets_kernel(P* buckets, unsigned N)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid < N) buckets[tid] = P::zero(); // zero point
    }

    // this kernel splits the scalars into digits of size c
    // each thread splits a single scalar into nof_bms digits
    template <typename S>
    __global__ void split_scalars_kernel(
      unsigned* buckets_indices,
      unsigned* point_indices,
      S* scalars,
      unsigned total_size,
      unsigned points_size,
      unsigned msm_size,
      unsigned nof_bms,
      unsigned bm_bitsize,
      unsigned c)
    {
      // constexpr unsigned sign_mask = 0x80000000;
      // constexpr unsigned trash_bucket = 0x80000000;
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      unsigned bucket_index;
      // unsigned bucket_index2;
      unsigned current_index;
      unsigned msm_index = tid / msm_size;
      // unsigned borrow = 0;
      if (tid < total_size) {
        S scalar = scalars[tid];
        for (unsigned bm = 0; bm < nof_bms; bm++) {
          bucket_index = scalar.get_scalar_digit(bm, c);
#ifdef SIGNED_DIG
          bucket_index += borrow;
          borrow = 0;
          unsigned sign = 0;
          if (bucket_index > (1 << (c - 1))) {
            bucket_index = (1 << c) - bucket_index;
            borrow = 1;
            sign = sign_mask;
          }
#endif
          current_index = bm * total_size + tid;
#ifdef SIGNED_DIG
          point_indices[current_index] = sign | tid; // the point index is saved for later
#else
          buckets_indices[current_index] =
            (msm_index << (c + bm_bitsize)) | (bm << c) |
            bucket_index; // the bucket module number and the msm number are appended at the msbs
          if (scalar == S::zero() || bucket_index == 0) buckets_indices[current_index] = 0; // will be skipped
          point_indices[current_index] = tid % points_size; // the point index is saved for later
#endif
        }
      }
    }

    template <typename S>
    __global__ void
    find_cutoff_kernel(unsigned* v, unsigned size, unsigned cutoff, unsigned run_length, unsigned* result)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned nof_threads = (size + run_length - 1) / run_length;
      if (tid >= nof_threads) { return; }
      const unsigned start_index = tid * run_length;
      for (int i = start_index; i < min(start_index + run_length, size - 1); i++) {
        if (v[i] > cutoff && v[i + 1] <= cutoff) {
          result[0] = i + 1;
          return;
        }
      }
      if (tid == 0 && v[size - 1] > cutoff) { result[0] = size; }
    }

    template <typename S>
    __global__ void
    find_max_size(unsigned* bucket_sizes, unsigned* single_bucket_indices, unsigned c, unsigned* largest_bucket_size)
    {
      for (int i = 0;; i++) {
        if (single_bucket_indices[i] & ((1 << c) - 1)) {
          largest_bucket_size[0] = bucket_sizes[i];
          largest_bucket_size[1] = i;
          break;
        }
      }
    }

    // this kernel adds up the points in each bucket
    template <typename P, typename A>
    __global__ void accumulate_buckets_kernel(
      P* __restrict__ buckets,
      unsigned* __restrict__ bucket_offsets,
      unsigned* __restrict__ bucket_sizes,
      unsigned* __restrict__ single_bucket_indices,
      const unsigned* __restrict__ point_indices,
      A* __restrict__ points,
      const unsigned nof_buckets,
      const unsigned nof_buckets_to_compute,
      const unsigned msm_idx_shift,
      const unsigned c)
    {
      // constexpr unsigned sign_mask = 0x80000000;
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_buckets_to_compute) return;
      if ((single_bucket_indices[tid] & ((1 << c) - 1)) == 0) {
        return; // skip zero buckets
      }
#ifdef SIGNED_DIG // todo - fix
      const unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
      const unsigned bm_index = (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1)) >> c;
      const unsigned bucket_index =
        msm_index * nof_buckets + bm_index * ((1 << (c - 1)) + 1) + (single_bucket_indices[tid] & ((1 << c) - 1));
#else
      unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
      unsigned bucket_index = msm_index * nof_buckets + (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1));
#endif
      const unsigned bucket_offset = bucket_offsets[tid];
      const unsigned bucket_size = bucket_sizes[tid];

      P bucket; // get rid of init buckets? no.. because what about buckets with no points
      for (unsigned i = 0; i < bucket_size;
           i++) { // add the relevant points starting from the relevant offset up to the bucket size
        unsigned point_ind = point_indices[bucket_offset + i];
#ifdef SIGNED_DIG
        unsigned sign = point_ind & sign_mask;
        point_ind &= ~sign_mask;
        A point = points[point_ind];
        if (sign) point = A::neg(point);
#else
        A point = points[point_ind];
#endif
        bucket = i ? bucket + point : P::from_affine(point);
      }
      buckets[bucket_index] = bucket;
    }

    template <typename P, typename A>
    __global__ void accumulate_large_buckets_kernel(
      P* __restrict__ buckets,
      unsigned* __restrict__ bucket_offsets,
      unsigned* __restrict__ bucket_sizes,
      unsigned* __restrict__ single_bucket_indices,
      const unsigned* __restrict__ point_indices,
      A* __restrict__ points,
      const unsigned nof_buckets,
      const unsigned nof_buckets_to_compute,
      const unsigned msm_idx_shift,
      const unsigned c,
      const unsigned threads_per_bucket,
      const unsigned max_run_length)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      unsigned large_bucket_index = tid / threads_per_bucket;
      unsigned bucket_segment_index = tid % threads_per_bucket;
      if (tid >= nof_buckets_to_compute * threads_per_bucket) { return; }
      if ((single_bucket_indices[large_bucket_index] & ((1 << c) - 1)) == 0) { // dont need
        return;                                                                // skip zero buckets
      }
      unsigned write_bucket_index = bucket_segment_index * nof_buckets_to_compute + large_bucket_index;
      const unsigned bucket_offset = bucket_offsets[large_bucket_index] + bucket_segment_index * max_run_length;
      const unsigned bucket_size = bucket_sizes[large_bucket_index] > bucket_segment_index * max_run_length
                                     ? bucket_sizes[large_bucket_index] - bucket_segment_index * max_run_length
                                     : 0;
      P bucket;
      unsigned run_length = min(bucket_size, max_run_length);
      for (unsigned i = 0; i < run_length;
           i++) { // add the relevant points starting from the relevant offset up to the bucket size
        unsigned point_ind = point_indices[bucket_offset + i];
        A point = points[point_ind];
        bucket = i ? bucket + point : P::from_affine(point); // init empty buckets
      }
      buckets[write_bucket_index] = run_length ? bucket : P::zero();
    }

    template <typename P>
    __global__ void
    distribute_large_buckets_kernel(P* large_buckets, P* buckets, unsigned* single_bucket_indices, unsigned size)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= size) { return; }
      buckets[single_bucket_indices[tid]] = large_buckets[tid];
    }

    // this kernel sums the entire bucket module
    // each thread deals with a single bucket module
    template <typename P>
    __global__ void big_triangle_sum_kernel(P* buckets, P* final_sums, unsigned nof_bms, unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_bms) return;
#ifdef SIGNED_DIG
      unsigned buckets_in_bm = (1 << c) + 1;
#else
      unsigned buckets_in_bm = (1 << c);
#endif
      P line_sum = buckets[(tid + 1) * buckets_in_bm - 1];
      final_sums[tid] = line_sum;
      for (unsigned i = buckets_in_bm - 2; i > 0; i--) {
        line_sum = line_sum + buckets[tid * buckets_in_bm + i]; // using the running sum method
        final_sums[tid] = final_sums[tid] + line_sum;
      }
    }

    // this kernel uses single scalar multiplication to multiply each bucket by its index
    // each thread deals with a single bucket
    template <typename P, typename S>
    __global__ void ssm_buckets_kernel(P* buckets, unsigned* single_bucket_indices, unsigned nof_buckets, unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid > nof_buckets) return;
      unsigned bucket_index = single_bucket_indices[tid];
      S scalar_bucket_multiplier;
      scalar_bucket_multiplier = {
        bucket_index & ((1 << c) - 1), 0, 0, 0, 0, 0, 0, 0}; // the index without the bucket module index
      buckets[bucket_index] = scalar_bucket_multiplier * buckets[bucket_index];
    }

    template <typename P>
    __global__ void last_pass_kernel(P* final_buckets, P* final_sums, unsigned num_sums)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid > num_sums) return;
      final_sums[tid] = final_buckets[2 * tid + 1];
    }

    // this kernel computes the final result using the double and add algorithm
    // it is done by a single thread
    template <typename P, typename S>
    __global__ void
    final_accumulation_kernel(P* final_sums, P* final_results, unsigned nof_msms, unsigned nof_bms, unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid > nof_msms) return;
      P final_result = P::zero();
      for (unsigned i = nof_bms; i > 1; i--) {
        final_result = final_result + final_sums[i - 1 + tid * nof_bms]; // add
        for (unsigned j = 0; j < c; j++)                                 // double
        {
          final_result = final_result + final_result;
        }
      }
      final_results[tid] = final_result + final_sums[tid * nof_bms];
    }

    // this function computes msm using the bucket method
    template <typename S, typename P, typename A>
    cudaError_t bucket_method_msm(
      int bitsize,
      int c,
      S* scalars,
      A* points,
      int size,
      P* final_result,
      bool are_scalars_on_device,
      bool are_scalars_montgomery_form,
      bool are_points_on_device,
      bool are_points_montgomery_form,
      bool is_result_on_device,
      bool is_big_triangle,
      int large_bucket_factor,
      bool is_async,
      cudaStream_t stream)
    {
      CHK_INIT_IF_RETURN();

      S* d_scalars;
      A* d_points;
      if (!are_scalars_on_device) {
        // copy scalars to gpu
        CHK_IF_RETURN(cudaMallocAsync(&d_scalars, sizeof(S) * size, stream));
        CHK_IF_RETURN(cudaMemcpyAsync(d_scalars, scalars, sizeof(S) * size, cudaMemcpyHostToDevice, stream));
      } else {
        d_scalars = scalars;
      }
      cudaStream_t stream_points;
      if (!are_points_on_device || are_points_montgomery_form) CHK_IF_RETURN(cudaStreamCreate(&stream_points));
      if (!are_points_on_device) {
        // copy points to gpu
        CHK_IF_RETURN(cudaMallocAsync(&d_points, sizeof(A) * size, stream_points));
        CHK_IF_RETURN(cudaMemcpyAsync(d_points, points, sizeof(A) * size, cudaMemcpyHostToDevice, stream_points));
      } else {
        d_points = points;
      }
      if (are_scalars_montgomery_form) {
        if (are_scalars_on_device) {
          S* d_mont_scalars;
          CHK_IF_RETURN(cudaMallocAsync(&d_mont_scalars, sizeof(S) * size, stream));
          CHK_IF_RETURN(mont::FromMontgomery(d_scalars, size, stream, d_mont_scalars));
          d_scalars = d_mont_scalars;
        } else
          CHK_IF_RETURN(mont::FromMontgomery(d_scalars, size, stream, d_scalars));
      }
      if (are_points_montgomery_form) {
        if (are_points_on_device) {
          A* d_mont_points;
          CHK_IF_RETURN(cudaMallocAsync(&d_mont_points, sizeof(A) * size, stream_points));
          CHK_IF_RETURN(mont::FromMontgomery(d_points, size, stream_points, d_mont_points));
          d_points = d_mont_points;
        } else
          CHK_IF_RETURN(mont::FromMontgomery(d_points, size, stream_points, d_points));
      }
      cudaEvent_t event_points_uploaded;
      if (!are_points_on_device || are_points_montgomery_form) {
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_points_uploaded, cudaEventDisableTiming));
        CHK_IF_RETURN(cudaEventRecord(event_points_uploaded, stream_points));
      }

      P* buckets;
      // compute number of bucket modules and number of buckets in each module
      unsigned nof_bms = (bitsize + c - 1) / c;
      unsigned bm_bitsize = (unsigned)ceil(log2(nof_bms));
#ifdef SIGNED_DIG
      unsigned nof_buckets = nof_bms * ((1 << (c - 1)) + 1); // signed digits
#else
      unsigned nof_buckets = nof_bms << c;
#endif
      CHK_IF_RETURN(cudaMallocAsync(&buckets, sizeof(P) * nof_buckets, stream));

      // launch the bucket initialization kernel with maximum threads
      unsigned NUM_THREADS = 1 << 10;
      unsigned NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
      initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, nof_buckets);

      unsigned* bucket_indices;
      unsigned* point_indices;
      CHK_IF_RETURN(cudaMallocAsync(&bucket_indices, sizeof(unsigned) * size * (nof_bms + 1), stream));
      CHK_IF_RETURN(cudaMallocAsync(&point_indices, sizeof(unsigned) * size * (nof_bms + 1), stream));

      // split scalars into digits
      NUM_THREADS = 1 << 10;
      NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;
      split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        bucket_indices + size, point_indices + size, d_scalars, size, size, size, nof_bms, bm_bitsize, c);
      //+size - leaving the first bm free for the out of place sort later

      // sort indices - the indices are sorted from smallest to largest in order to group together the points that
      // belong to each bucket
      unsigned* sort_indices_temp_storage{};
      size_t sort_indices_temp_storage_bytes;
      // The second to last parameter is the default value supplied explicitly to allow passing the stream
      // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for
      // more info
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + size, bucket_indices,
        point_indices + size, point_indices, size, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
      for (unsigned i = 0; i < nof_bms; i++) {
        unsigned offset_out = i * size;
        unsigned offset_in = offset_out + size;
        // The second to last parameter is the default value supplied explicitly to allow passing the stream
        // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for
        // more info
        CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
          sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
          bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, size, 0,
          sizeof(unsigned) * 8, stream));
      }
      CHK_IF_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));

      // find bucket_sizes
      unsigned* single_bucket_indices;
      unsigned* bucket_sizes;
      unsigned* nof_buckets_to_compute;
      CHK_IF_RETURN(cudaMallocAsync(&single_bucket_indices, sizeof(unsigned) * nof_buckets, stream));
      CHK_IF_RETURN(cudaMallocAsync(&bucket_sizes, sizeof(unsigned) * nof_buckets, stream));
      CHK_IF_RETURN(cudaMallocAsync(&nof_buckets_to_compute, sizeof(unsigned), stream));
      unsigned* encode_temp_storage{};
      size_t encode_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
        nof_buckets_to_compute, nof_bms * size, stream));
      CHK_IF_RETURN(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
        nof_buckets_to_compute, nof_bms * size, stream));
      CHK_IF_RETURN(cudaFreeAsync(encode_temp_storage, stream));

      // get offsets - where does each new bucket begin
      unsigned* bucket_offsets;
      CHK_IF_RETURN(cudaMallocAsync(&bucket_offsets, sizeof(unsigned) * nof_buckets, stream));
      unsigned* offsets_temp_storage{};
      size_t offsets_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets, stream));
      CHK_IF_RETURN(cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets, stream));
      CHK_IF_RETURN(cudaFreeAsync(offsets_temp_storage, stream));

      // sort by bucket sizes
      unsigned h_nof_buckets_to_compute;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &h_nof_buckets_to_compute, nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));

      // if all points are 0 just return point 0
      if (h_nof_buckets_to_compute == 0) {
        if (!is_result_on_device)
          final_result[0] = P::zero();
        else {
          P* h_final_result = (P*)malloc(sizeof(P));
          h_final_result[0] = P::zero();
          CHK_IF_RETURN(cudaMemcpyAsync(final_result, h_final_result, sizeof(P), cudaMemcpyHostToDevice, stream));
        }

        return CHK_LAST();
      }

      unsigned* sorted_bucket_sizes;
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_sizes, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      unsigned* sorted_bucket_offsets;
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_offsets, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      unsigned* sort_offsets_temp_storage{};
      size_t sort_offsets_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes, sorted_bucket_sizes, bucket_offsets,
        sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes, sorted_bucket_sizes, bucket_offsets,
        sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_offsets_temp_storage, stream));

      unsigned* sorted_single_bucket_indices;
      CHK_IF_RETURN(
        cudaMallocAsync(&sorted_single_bucket_indices, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      unsigned* sort_single_temp_storage{};
      size_t sort_single_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes, sorted_bucket_sizes,
        single_bucket_indices, sorted_single_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8,
        stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_single_temp_storage, sort_single_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes, sorted_bucket_sizes,
        single_bucket_indices, sorted_single_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8,
        stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_single_temp_storage, stream));

      // find large buckets
      unsigned avarage_size = size / (1 << c);
      unsigned bucket_th = large_bucket_factor * avarage_size;
      unsigned* nof_large_buckets;
      CHK_IF_RETURN(cudaMallocAsync(&nof_large_buckets, sizeof(unsigned), stream));
      CHK_IF_RETURN(cudaMemset(nof_large_buckets, 0, sizeof(unsigned)));

      unsigned TOTAL_THREADS = 129000; // todo - device dependant
      unsigned cutoff_run_length = max(2, h_nof_buckets_to_compute / TOTAL_THREADS);
      unsigned cutoff_nof_runs = (h_nof_buckets_to_compute + cutoff_run_length - 1) / cutoff_run_length;
      NUM_THREADS = min(1 << 5, cutoff_nof_runs);
      NUM_BLOCKS = (cutoff_nof_runs + NUM_THREADS - 1) / NUM_THREADS;
      find_cutoff_kernel<S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        sorted_bucket_sizes, h_nof_buckets_to_compute, bucket_th, cutoff_run_length, nof_large_buckets);

      unsigned h_nof_large_buckets;
      CHK_IF_RETURN(
        cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));

      unsigned* max_res;
      CHK_IF_RETURN(cudaMallocAsync(&max_res, sizeof(unsigned) * 2, stream));
      find_max_size<S><<<1, 1, 0, stream>>>(sorted_bucket_sizes, sorted_single_bucket_indices, c, max_res);

      unsigned h_max_res[2];
      CHK_IF_RETURN(cudaMemcpyAsync(h_max_res, max_res, sizeof(unsigned) * 2, cudaMemcpyDeviceToHost, stream));
      unsigned h_largest_bucket_size = h_max_res[0];
      unsigned h_nof_zero_large_buckets = h_max_res[1];
      unsigned large_buckets_to_compute =
        h_nof_large_buckets > h_nof_zero_large_buckets ? h_nof_large_buckets - h_nof_zero_large_buckets : 0;

      if (!are_points_on_device || are_points_montgomery_form) {
        // by this point, points need to be already uploaded and un-Montgomeried
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_points_uploaded));
        CHK_IF_RETURN(cudaStreamDestroy(stream_points));
      }

      cudaStream_t stream_large_buckets;
      cudaEvent_t event_large_buckets_accumulated;
      P* large_buckets;
      if (large_buckets_to_compute > 0 && bucket_th > 0) {
        CHK_IF_RETURN(cudaStreamCreate(&stream_large_buckets));
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_large_buckets_accumulated, cudaEventDisableTiming));

        unsigned threads_per_bucket =
          1 << (unsigned)ceil(log2((h_largest_bucket_size + bucket_th - 1) / bucket_th)); // global param
        unsigned max_bucket_size_run_length = (h_largest_bucket_size + threads_per_bucket - 1) / threads_per_bucket;
        unsigned total_large_buckets_size = large_buckets_to_compute * threads_per_bucket;
        CHK_IF_RETURN(cudaMallocAsync(&large_buckets, sizeof(P) * total_large_buckets_size, stream));

        NUM_THREADS = min(1 << 8, total_large_buckets_size);
        NUM_BLOCKS = (total_large_buckets_size + NUM_THREADS - 1) / NUM_THREADS;
        accumulate_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          large_buckets, sorted_bucket_offsets + h_nof_zero_large_buckets,
          sorted_bucket_sizes + h_nof_zero_large_buckets, sorted_single_bucket_indices + h_nof_zero_large_buckets,
          point_indices, d_points, nof_buckets, large_buckets_to_compute, c + bm_bitsize, c, threads_per_bucket,
          max_bucket_size_run_length);

        // reduce
        for (int s = total_large_buckets_size >> 1; s > large_buckets_to_compute - 1; s >>= 1) {
          NUM_THREADS = min(MAX_TH, s);
          NUM_BLOCKS = (s + NUM_THREADS - 1) / NUM_THREADS;
          single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
            large_buckets, large_buckets, s * 2, 0, 0, 0, s);
        }

        // distribute
        NUM_THREADS = min(MAX_TH, large_buckets_to_compute);
        NUM_BLOCKS = (large_buckets_to_compute + NUM_THREADS - 1) / NUM_THREADS;
        distribute_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          large_buckets, buckets, sorted_single_bucket_indices + h_nof_zero_large_buckets, large_buckets_to_compute);

        CHK_IF_RETURN(cudaEventRecord(event_large_buckets_accumulated, stream_large_buckets));
        CHK_IF_RETURN(cudaStreamDestroy(stream_large_buckets));
      } else {
        h_nof_large_buckets = 0;
      }

      // launch the accumulation kernel with maximum threads
      if (h_nof_buckets_to_compute > h_nof_large_buckets) {
        NUM_THREADS = 1 << 8;
        NUM_BLOCKS = (h_nof_buckets_to_compute - h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
          buckets, sorted_bucket_offsets + h_nof_large_buckets, sorted_bucket_sizes + h_nof_large_buckets,
          sorted_single_bucket_indices + h_nof_large_buckets, point_indices, d_points, nof_buckets,
          h_nof_buckets_to_compute - h_nof_large_buckets, c + bm_bitsize, c);
      }

      if (large_buckets_to_compute > 0 && bucket_th > 0)
        // all the large buckets need to be accumulated before the final summation
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_large_buckets_accumulated));

#ifdef SSM_SUM
      // sum each bucket
      NUM_THREADS = 1 << 10;
      NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
      ssm_buckets_kernel<fake_point, fake_scalar>
        <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, single_bucket_indices, nof_buckets, c);

      // sum each bucket module
      P* final_results;
      CHK_IF_RETURN(cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream));
      NUM_THREADS = 1 << c;
      NUM_BLOCKS = nof_bms;
      sum_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, final_results);
#endif

      P* d_final_result;
      if (!is_result_on_device) CHK_IF_RETURN(cudaMallocAsync(&d_final_result, sizeof(P), stream));

      P* final_results;
      if (is_big_triangle) {
        CHK_IF_RETURN(cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream));
        // launch the bucket module sum kernel - a thread for each bucket module
        NUM_THREADS = nof_bms;
        NUM_BLOCKS = 1;
#ifdef SIGNED_DIG
        big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
          buckets, final_results, nof_bms, c - 1); // sighed digits
#else
        big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, final_results, nof_bms, c);
#endif
      } else {
        unsigned source_bits_count = c;
        // bool odd_source_c = source_bits_count % 2;
        unsigned source_windows_count = nof_bms;
        unsigned source_buckets_count = nof_buckets;
        P* source_buckets = buckets;
        buckets = nullptr;
        P* target_buckets;
        P* temp_buckets1;
        P* temp_buckets2;
        for (unsigned i = 0;; i++) {
          const unsigned target_bits_count = (source_bits_count + 1) >> 1;                 // c/2=8
          const unsigned target_windows_count = source_windows_count << 1;                 // nof bms*2 = 32
          const unsigned target_buckets_count = target_windows_count << target_bits_count; // bms*2^c = 32*2^8
          CHK_IF_RETURN(
            cudaMallocAsync(&target_buckets, sizeof(P) * target_buckets_count, stream)); // 32*2^8*2^7 buckets
          CHK_IF_RETURN(
            cudaMallocAsync(&temp_buckets1, sizeof(P) * source_buckets_count / 2, stream)); // 32*2^8*2^7 buckets
          CHK_IF_RETURN(
            cudaMallocAsync(&temp_buckets2, sizeof(P) * source_buckets_count / 2, stream)); // 32*2^8*2^7 buckets

          if (source_bits_count > 0) {
            for (unsigned j = 0; j < target_bits_count; j++) {
              // unsigned last_j = target_bits_count - 1;
              unsigned nof_threads = (source_buckets_count >> (1 + j));
              NUM_THREADS = min(MAX_TH, nof_threads);
              NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
              single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                j == 0 ? source_buckets : temp_buckets1, j == target_bits_count - 1 ? target_buckets : temp_buckets1,
                1 << (source_bits_count - j), j == target_bits_count - 1 ? 1 << target_bits_count : 0, 0, 0,
                nof_threads);

              NUM_THREADS = min(MAX_TH, nof_threads);
              NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
              single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                j == 0 ? source_buckets : temp_buckets2, j == target_bits_count - 1 ? target_buckets : temp_buckets2,
                1 << (target_bits_count - j), j == target_bits_count - 1 ? 1 << target_bits_count : 0, 1, 0,
                nof_threads);
            }
          }
          if (target_bits_count == 1) {
            nof_bms = bitsize;
            CHK_IF_RETURN(cudaMallocAsync(&final_results, sizeof(P) * nof_bms, stream));
            NUM_THREADS = 32;
            NUM_BLOCKS = (nof_bms + NUM_THREADS - 1) / NUM_THREADS;
            last_pass_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(target_buckets, final_results, nof_bms);
            c = 1;
            CHK_IF_RETURN(cudaFreeAsync(source_buckets, stream));
            CHK_IF_RETURN(cudaFreeAsync(target_buckets, stream));
            CHK_IF_RETURN(cudaFreeAsync(temp_buckets1, stream));
            CHK_IF_RETURN(cudaFreeAsync(temp_buckets2, stream));
            break;
          }
          CHK_IF_RETURN(cudaFreeAsync(source_buckets, stream));
          CHK_IF_RETURN(cudaFreeAsync(temp_buckets1, stream));
          CHK_IF_RETURN(cudaFreeAsync(temp_buckets2, stream));
          source_buckets = target_buckets;
          target_buckets = nullptr;
          temp_buckets1 = nullptr;
          temp_buckets2 = nullptr;
          source_bits_count = target_bits_count;
          // odd_source_c = source_bits_count % 2;
          source_windows_count = target_windows_count;
          source_buckets_count = target_buckets_count;
        }
      }

      // launch the double and add kernel, a single thread
      final_accumulation_kernel<P, S>
        <<<1, 1, 0, stream>>>(final_results, is_result_on_device ? final_result : d_final_result, 1, nof_bms, c);
      CHK_IF_RETURN(cudaFreeAsync(final_results, stream));

      if (!is_result_on_device)
        CHK_IF_RETURN(cudaMemcpyAsync(final_result, d_final_result, sizeof(P), cudaMemcpyDeviceToHost, stream));

      // free memory
      if (!are_scalars_on_device) CHK_IF_RETURN(cudaFreeAsync(d_scalars, stream));
      if (!are_points_on_device) CHK_IF_RETURN(cudaFreeAsync(d_points, stream));
      if (!is_result_on_device) CHK_IF_RETURN(cudaFreeAsync(d_final_result, stream));
      CHK_IF_RETURN(cudaFreeAsync(buckets, stream));
#ifndef PHASE1_TEST
      CHK_IF_RETURN(cudaFreeAsync(bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(point_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(single_bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_sizes, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_buckets_to_compute, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_offsets, stream));
#endif
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_sizes, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_offsets, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_single_bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_large_buckets, stream));
      CHK_IF_RETURN(cudaFreeAsync(max_res, stream));
      if (large_buckets_to_compute > 0 && bucket_th > 0) CHK_IF_RETURN(cudaFreeAsync(large_buckets, stream));

      if (!is_async) CHK_IF_RETURN(cudaStreamSynchronize(stream));

      return CHK_LAST();
    }

    // this function computes multiple msms using the bucket method
    template <typename S, typename P, typename A>
    cudaError_t batched_bucket_method_msm(
      unsigned bitsize,
      unsigned c,
      S* scalars,
      A* points,
      unsigned batch_size,
      unsigned msm_size,
      unsigned points_size,
      P* final_results,
      bool are_scalars_on_device,
      bool are_scalars_montgomery_form,
      bool are_points_on_device,
      bool are_points_montgomery_form,
      bool are_results_on_device,
      bool is_async,
      cudaStream_t stream)
    {
      CHK_INIT_IF_RETURN();

      unsigned total_size = batch_size * msm_size;
      S* d_scalars;
      A* d_points;
      if (!are_scalars_on_device) {
        // copy scalars to gpu
        CHK_IF_RETURN(cudaMallocAsync(&d_scalars, sizeof(S) * total_size, stream));
        CHK_IF_RETURN(cudaMemcpyAsync(d_scalars, scalars, sizeof(S) * total_size, cudaMemcpyHostToDevice, stream));
      } else {
        d_scalars = scalars;
      }
      cudaStream_t stream_points;
      if (!are_points_on_device || are_points_montgomery_form) cudaStreamCreate(&stream_points);
      if (!are_points_on_device) {
        // copy points to gpu
        cudaMallocAsync(&d_points, sizeof(A) * points_size, stream_points);
        cudaMemcpyAsync(d_points, points, sizeof(A) * points_size, cudaMemcpyHostToDevice, stream_points);
      } else {
        d_points = points;
      }
      if (are_scalars_montgomery_form) {
        if (are_scalars_on_device) {
          S* d_mont_scalars;
          cudaMallocAsync(&d_mont_scalars, sizeof(S) * total_size, stream);
          mont::FromMontgomery(d_scalars, total_size, stream, d_mont_scalars);
          d_scalars = d_mont_scalars;
        } else
          mont::FromMontgomery(d_scalars, total_size, stream, d_scalars);
      }
      if (are_points_montgomery_form) {
        if (are_points_on_device) {
          A* d_mont_points;
          cudaMallocAsync(&d_mont_points, sizeof(A) * points_size, stream_points);
          mont::FromMontgomery(d_points, points_size, stream_points, d_mont_points);
          d_points = d_mont_points;
        } else
          mont::FromMontgomery(d_points, points_size, stream_points, d_points);
      }
      cudaEvent_t event_points_uploaded;
      if (!are_points_on_device || are_points_montgomery_form) {
        cudaEventCreateWithFlags(&event_points_uploaded, cudaEventDisableTiming);
        cudaEventRecord(event_points_uploaded, stream_points);
      }

      P* buckets;
      // compute number of bucket modules and number of buckets in each module
      unsigned nof_bms = (bitsize + c - 1) / c;
      unsigned bm_bitsize = (unsigned)ceil(log2(nof_bms));
      unsigned nof_buckets = (nof_bms << c);
      unsigned total_nof_buckets = nof_buckets * batch_size;
      CHK_IF_RETURN(cudaMallocAsync(&buckets, sizeof(P) * total_nof_buckets, stream));

      // lanch the bucket initialization kernel with maximum threads
      unsigned NUM_THREADS = 1 << 10;
      unsigned NUM_BLOCKS = (total_nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
      initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, total_nof_buckets);

      unsigned* bucket_indices;
      unsigned* point_indices;
      CHK_IF_RETURN(cudaMallocAsync(&bucket_indices, sizeof(unsigned) * total_size * nof_bms, stream));
      CHK_IF_RETURN(cudaMallocAsync(&point_indices, sizeof(unsigned) * total_size * nof_bms, stream));

      // split scalars into digits
      NUM_THREADS = 1 << 10;
      NUM_BLOCKS = (total_size + NUM_THREADS - 1) / NUM_THREADS;
      split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        bucket_indices, point_indices, d_scalars, total_size, points_size, msm_size, nof_bms, bm_bitsize, c);

      // sort indices - the indices are sorted from smallest to largest in order to group together the points that
      // belong to each bucket
      unsigned* sorted_bucket_indices;
      unsigned* sorted_point_indices;
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_indices, sizeof(unsigned) * (total_size * nof_bms), stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_point_indices, sizeof(unsigned) * (total_size * nof_bms), stream));

      unsigned* sort_indices_temp_storage{};
      size_t sort_indices_temp_storage_bytes;
      // The second to last parameter is the default value supplied explicitly to allow passing the stream
      // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for
      // more info
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, total_size * nof_bms, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
      // The second to last parameter is the default value supplied explicitly to allow passing the stream
      // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for
      // more info
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, total_size * nof_bms, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));

      // find bucket_sizes
      unsigned* single_bucket_indices;
      unsigned* bucket_sizes;
      unsigned* total_nof_buckets_to_compute;
      CHK_IF_RETURN(cudaMallocAsync(&single_bucket_indices, sizeof(unsigned) * total_nof_buckets, stream));
      CHK_IF_RETURN(cudaMallocAsync(&bucket_sizes, sizeof(unsigned) * total_nof_buckets, stream));
      CHK_IF_RETURN(cudaMallocAsync(&total_nof_buckets_to_compute, sizeof(unsigned), stream));
      unsigned* encode_temp_storage{};
      size_t encode_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
        total_nof_buckets_to_compute, nof_bms * total_size, stream));
      CHK_IF_RETURN(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
        total_nof_buckets_to_compute, nof_bms * total_size, stream));
      CHK_IF_RETURN(cudaFreeAsync(encode_temp_storage, stream));

      // get offsets - where does each new bucket begin
      unsigned* bucket_offsets;
      CHK_IF_RETURN(cudaMallocAsync(&bucket_offsets, sizeof(unsigned) * total_nof_buckets, stream));
      unsigned* offsets_temp_storage{};
      size_t offsets_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets, stream));
      CHK_IF_RETURN(cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets, stream));
      CHK_IF_RETURN(cudaFreeAsync(offsets_temp_storage, stream));

      unsigned h_nof_buckets_to_compute;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &h_nof_buckets_to_compute, total_nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));

      if (!are_points_on_device || are_points_montgomery_form) {
        // by this point, points need to be already uploaded and un-Montgomeried
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_points_uploaded));
        CHK_IF_RETURN(cudaStreamDestroy(stream_points));
      }

      // launch the accumulation kernel with maximum threads
      NUM_THREADS = 1 << 8;
      NUM_BLOCKS = (total_nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
      accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        buckets, bucket_offsets, bucket_sizes, single_bucket_indices, sorted_point_indices, d_points, nof_buckets,
        h_nof_buckets_to_compute, c + bm_bitsize, c);

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
      CHK_IF_RETURN(cudaMallocAsync(&bm_sums, sizeof(P) * nof_bms * batch_size, stream));
      // launch the bucket module sum kernel - a thread for each bucket module
      NUM_THREADS = 1 << 8;
      NUM_BLOCKS = (nof_bms * batch_size + NUM_THREADS - 1) / NUM_THREADS;
      big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, bm_sums, nof_bms * batch_size, c);
      // #endif

      P* d_final_results;
      if (!are_results_on_device) CHK_IF_RETURN(cudaMallocAsync(&d_final_results, sizeof(P) * batch_size, stream));

      // launch the double and add kernel, a single thread for each msm
      NUM_THREADS = 1 << 8;
      NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS;
      final_accumulation_kernel<P, S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        bm_sums, are_results_on_device ? final_results : d_final_results, batch_size, nof_bms, c);

      // copy final result to host
      if (!are_results_on_device)
        CHK_IF_RETURN(
          cudaMemcpyAsync(final_results, d_final_results, sizeof(P) * batch_size, cudaMemcpyDeviceToHost, stream));

      // free memory
      if (!are_scalars_on_device) CHK_IF_RETURN(cudaFreeAsync(d_scalars, stream));
      if (!are_points_on_device) CHK_IF_RETURN(cudaFreeAsync(d_points, stream));
      if (!are_results_on_device) CHK_IF_RETURN(cudaFreeAsync(d_final_results, stream));
      CHK_IF_RETURN(cudaFreeAsync(buckets, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(point_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_point_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(single_bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_sizes, stream));
      CHK_IF_RETURN(cudaFreeAsync(total_nof_buckets_to_compute, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_offsets, stream));
      CHK_IF_RETURN(cudaFreeAsync(bm_sums, stream));

      if (is_async) return CHK_LAST();
      return CHK_STICKY(cudaStreamSynchronize(stream));
    }

  } // namespace

  template <typename A>
  MSMConfig DefaultMSMConfig()
  {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
    MSMConfig config = {
      ctx,   // ctx
      0,     // points_size
      1,     // precompute_factor
      0,     // c
      0,     // bitsize
      10,    // large_bucket_factor
      1,     // batch_size
      false, // are_scalars_on_device
      false, // are_scalars_montgomery_form
      false, // are_points_on_device
      false, // are_points_montgomery_form
      false, // are_results_on_device
      false, // is_big_triangle
      false, // is_async
    };
    return config;
  }

  template <typename S, typename A, typename P>
  cudaError_t MSM(S* scalars, A* points, int msm_size, MSMConfig& config, P* results)
  {
    int bitsize = (config.bitsize == 0) ? S::NBITS : config.bitsize;
    cudaStream_t& stream = config.ctx.stream;
    // TODO: DmytroTym/HadarIngonyama - unify the implementation of the bucket method and the batched bucket method in
    // one function
    if (config.batch_size == 1)
      return CHK_STICKY(bucket_method_msm(
        bitsize, 16, scalars, points, msm_size, results, config.are_scalars_on_device,
        config.are_scalars_montgomery_form, config.are_points_on_device, config.are_points_montgomery_form,
        config.are_results_on_device, config.is_big_triangle, config.large_bucket_factor, config.is_async, stream));
    else
      return CHK_STICKY(batched_bucket_method_msm(
        bitsize, (config.c == 0) ? get_optimal_c<S>(msm_size) : config.c, scalars, points, config.batch_size, msm_size,
        (config.points_size == 0) ? msm_size : config.points_size, results, config.are_scalars_on_device,
        config.are_scalars_montgomery_form, config.are_points_on_device, config.are_points_montgomery_form,
        config.are_results_on_device, config.is_async, stream));
  }

  /**
   * Extern "C" version of [MSM](@ref MSM) function with the following values of template parameters
   * (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` is the [scalar field](@ref scalar_t) of the curve;
   *  - `A` is the [affine representation](@ref affine_t) of curve points;
   *  - `P` is the [projective representation](@ref projective_t) of curve points.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, MSMCuda)(
    curve_config::scalar_t* scalars,
    curve_config::affine_t* points,
    int msm_size,
    MSMConfig& config,
    curve_config::projective_t* out)
  {
    return MSM<curve_config::scalar_t, curve_config::affine_t, curve_config::projective_t>(
      scalars, points, msm_size, config, out);
  }

  /**
   * Extern "C" version of [DefaultMSMConfig](@ref DefaultMSMConfig) function.
   */
  extern "C" MSMConfig CONCAT_EXPAND(CURVE, DefaultMSMConfig)() {
    return DefaultMSMConfig<curve_config::affine_t>();
  }

#if defined(G2_DEFINED)

  /**
   * Extern "C" version of [MSM](@ref MSM) function with the following values of template parameters
   * (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` is the [scalar field](@ref scalar_t) of the curve;
   *  - `A` is the [affine representation](@ref g2_affine_t) of G2 curve points;
   *  - `P` is the [projective representation](@ref g2_projective_t) of G2 curve points.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2MSMCuda)(
    curve_config::scalar_t* scalars,
    curve_config::g2_affine_t* points,
    int msm_size,
    MSMConfig& config,
    curve_config::g2_projective_t* out)
  {
    return MSM<curve_config::scalar_t, curve_config::g2_affine_t, curve_config::g2_projective_t>(
      scalars, points, msm_size, config, out);
  }

  /**
   * Extern "C" version of [DefaultMSMConfig](@ref DefaultMSMConfig) function for the G2 curve 
   * (functionally no different than the default MSM config function for G1).
   */
  extern "C" MSMConfig CONCAT_EXPAND(CURVE, G2DefaultMSMConfig)() {
    return DefaultMSMConfig<curve_config::g2_affine_t>();
  }

#endif

} // namespace msm