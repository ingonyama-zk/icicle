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
#include "utils/error_handler.cuh"
#include "utils/mont.cuh"
#include "utils/utils.h"
#include "utils/vec_ops.cuh"

namespace msm {

  namespace {

#define MAX_TH 256

    // #define SIGNED_DIG //WIP
    // #define SSM_SUM  //WIP

    template <typename A, typename P>
    __global__ void left_shift_kernel(A* points, const unsigned shift, const unsigned count, A* points_out)
    {
      const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= count) return;
      P point = P::from_affine(points[tid]);
      for (unsigned i = 0; i < shift; i++)
        point = P::dbl(point);
      points_out[tid] = P::to_affine(point);
    }

    unsigned get_optimal_c(int bitsize) { return max((unsigned)ceil(log2(bitsize)) - 4, 1U); }

    template <typename E>
    __global__ void normalize_kernel(E* inout, E factor, int n)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) inout[tid] = (inout[tid] + factor - 1) / factor;
    }

    // a kernel that writes to bucket_indices which enables large bucket accumulation to happen afterwards.
    // specifically we map thread indices to buckets which said threads will handle in accumulation.
    template <typename P>
    __global__ void initialize_large_bucket_indices(
      unsigned* sorted_bucket_sizes_sum,
      unsigned nof_pts_per_thread,
      unsigned nof_large_buckets,
      // log_nof_buckets_to_compute should be equal to ceil(log(nof_buckets_to_compute))
      unsigned log_nof_large_buckets,
      unsigned* bucket_indices)
    {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= nof_large_buckets) { return; }
      unsigned start = (sorted_bucket_sizes_sum[tid] + nof_pts_per_thread - 1) / nof_pts_per_thread + tid;
      unsigned end = (sorted_bucket_sizes_sum[tid + 1] + nof_pts_per_thread - 1) / nof_pts_per_thread + tid + 1;
      for (unsigned i = start; i < end; i++) {
        // this just concatenates two pieces of data - large bucket index and (i - start)
        bucket_indices[i] = tid | ((i - start) << log_nof_large_buckets);
      }
    }

    // this function provides a single step of reduction across buckets sizes of
    // which are given by large_bucket_sizes pointer
    template <typename P>
    __global__ void sum_reduction_variable_size_kernel(
      P* v,
      unsigned* bucket_sizes_sum,
      unsigned* bucket_sizes,
      unsigned* large_bucket_thread_indices,
      unsigned num_of_threads)
    {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= num_of_threads) { return; }

      unsigned large_bucket_tid = large_bucket_thread_indices[tid];
      unsigned segment_ind = tid - bucket_sizes_sum[large_bucket_tid] - large_bucket_tid;
      unsigned large_bucket_size = bucket_sizes[large_bucket_tid];
      if (segment_ind < (large_bucket_size >> 1)) { v[tid] = v[tid] + v[tid + ((large_bucket_size + 1) >> 1)]; }
    }

    template <typename P>
    __global__ void single_stage_multi_reduction_kernel(
      const P* v,
      P* v_r,
      unsigned block_size,
      unsigned write_stride,
      unsigned write_phase,
      unsigned step,
      unsigned num_of_threads)
    {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= num_of_threads) { return; }

      // we need shifted tid because we don't want to be reducing into zero buckets, this allows to skip them.
      // for write_phase==1, the read pattern is different so we don't skip over anything.
      const int shifted_tid = write_phase ? tid : tid + (tid + step) / step;
      const int jump = block_size / 2;
      const int block_id = shifted_tid / jump;
      // here the reason for shifting is the same as for shifted_tid but we skip over entire blocks which happens
      // only for write_phase=1 because of its read pattern.
      const int shifted_block_id = write_phase ? block_id + (block_id + step) / step : block_id;
      const int block_tid = shifted_tid % jump;
      const unsigned read_ind = block_size * shifted_block_id + block_tid;
      const unsigned write_ind = jump * shifted_block_id + block_tid;
      const unsigned v_r_key =
        write_stride ? ((write_ind / write_stride) * 2 + write_phase) * write_stride + write_ind % write_stride
                     : write_ind;
      v_r[v_r_key] = v[read_ind] + v[read_ind + jump];
    }

    // this kernel performs single scalar multiplication
    // each thread multiplies a single scalar and point
    template <typename P, typename S>
    __global__ void ssm_kernel(const S* scalars, const P* points, P* results, unsigned N)
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
      unsigned nof_scalars,
      unsigned points_size,
      unsigned msm_size,
      unsigned nof_bms,
      unsigned bm_bitsize,
      unsigned c,
      unsigned precomputed_bms_stride)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_scalars) return;

      unsigned bucket_index;
      unsigned current_index;
      unsigned msm_index = tid / msm_size;
      S& scalar = scalars[tid];
      for (unsigned bm = 0; bm < nof_bms; bm++) {
        const unsigned precomputed_index = bm / precomputed_bms_stride;
        const unsigned target_bm = bm % precomputed_bms_stride;

        bucket_index = scalar.get_scalar_digit(bm, c);
        current_index = bm * nof_scalars + tid;

        if (bucket_index != 0) {
          buckets_indices[current_index] =
            (msm_index << (c + bm_bitsize)) | (target_bm << c) |
            bucket_index; // the bucket module number and the msm number are appended at the msbs
        } else {
          buckets_indices[current_index] = 0; // will be skipped
        }
        point_indices[current_index] =
          tid % points_size + points_size * precomputed_index; // the point index is saved for later
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
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_buckets_to_compute) return;
      unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
      const unsigned single_bucket_index = (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1));
      unsigned bucket_index = msm_index * nof_buckets + single_bucket_index;
      const unsigned bucket_offset = bucket_offsets[tid];
      const unsigned bucket_size = bucket_sizes[tid];

      P bucket; // get rid of init buckets? no.. because what about buckets with no points
      for (unsigned i = 0; i < bucket_size;
           i++) { // add the relevant points starting from the relevant offset up to the bucket size
        unsigned point_ind = point_indices[bucket_offset + i];
        A point = points[point_ind];
        bucket =
          i ? (point == A::zero() ? bucket : bucket + point) : (point == A::zero() ? P::zero() : P::from_affine(point));
      }
      buckets[bucket_index] = bucket;
    }

    template <typename P, typename A>
    __global__ void accumulate_large_buckets_kernel(
      P* __restrict__ buckets,
      unsigned* __restrict__ bucket_offsets,
      unsigned* __restrict__ bucket_sizes,
      unsigned* __restrict__ large_bucket_thread_indices,
      unsigned* __restrict__ point_indices,
      A* __restrict__ points,
      const unsigned nof_buckets_to_compute,
      const unsigned c,
      const int points_per_thread,
      // log_nof_buckets_to_compute should be equal to ceil(log(nof_buckets_to_compute))
      const unsigned log_nof_buckets_to_compute,
      const unsigned nof_threads)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_threads) return;
      int bucket_segment_index = large_bucket_thread_indices[tid] >> log_nof_buckets_to_compute;
      large_bucket_thread_indices[tid] &= ((1 << log_nof_buckets_to_compute) - 1);
      int bucket_ind = large_bucket_thread_indices[tid];
      const unsigned bucket_offset = bucket_offsets[bucket_ind] + bucket_segment_index * points_per_thread;
      const unsigned bucket_size = max(0, (int)bucket_sizes[bucket_ind] - bucket_segment_index * points_per_thread);
      P bucket;
      unsigned run_length = min(bucket_size, points_per_thread);
      for (unsigned i = 0; i < run_length;
           i++) { // add the relevant points starting from the relevant offset up to the bucket size
        unsigned point_ind = point_indices[bucket_offset + i];
        A point = points[point_ind];
        bucket =
          i ? (point == A::zero() ? bucket : bucket + point) : (point == A::zero() ? P::zero() : P::from_affine(point));
      }
      buckets[tid] = run_length ? bucket : P::zero();
    }

    template <typename P>
    __global__ void distribute_large_buckets_kernel(
      const P* large_buckets,
      P* buckets,
      const unsigned* sorted_bucket_sizes_sum,
      const unsigned* single_bucket_indices,
      const unsigned size,
      const unsigned nof_buckets,
      const unsigned msm_idx_shift)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= size) { return; }

      unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
      unsigned bucket_index = msm_index * nof_buckets + (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1));
      unsigned large_bucket_index = sorted_bucket_sizes_sum[tid] + tid;
      buckets[bucket_index] = large_buckets[large_bucket_index];
    }

    // this kernel sums the entire bucket module
    // each thread deals with a single bucket module
    template <typename P>
    __global__ void big_triangle_sum_kernel(const P* buckets, P* final_sums, unsigned nof_bms, unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_bms) return;
      unsigned buckets_in_bm = (1 << c);
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
      if (tid >= nof_buckets) return;
      unsigned bucket_index = single_bucket_indices[tid];
      S scalar_bucket_multiplier;
      scalar_bucket_multiplier = {
        bucket_index & ((1 << c) - 1), 0, 0, 0, 0, 0, 0, 0}; // the index without the bucket module index
      buckets[bucket_index] = scalar_bucket_multiplier * buckets[bucket_index];
    }

    template <typename P>
    __global__ void last_pass_kernel(const P* final_buckets, P* final_sums, unsigned num_sums)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= num_sums) return;
      final_sums[tid] = final_buckets[2 * tid + 1];
    }

    // this kernel computes the final result using the double and add algorithm
    // it is done by a single thread
    template <typename P, typename S>
    __global__ void final_accumulation_kernel(
      const P* final_sums, P* final_results, unsigned nof_msms, unsigned nof_bms, unsigned nof_empty_bms, unsigned c)
    {
      unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (tid >= nof_msms) return;
      P final_result = P::zero();
      // Note: in some cases accumulation of bm is implemented such that some bms are known to be empty. Therefore
      // skipping them.
      for (unsigned i = nof_bms - nof_empty_bms; i > 1; i--) {
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
      unsigned bitsize,
      unsigned c,
      S* scalars,
      A* points,
      unsigned batch_size,      // number of MSMs to compute
      unsigned single_msm_size, // number of elements per MSM (a.k.a N)
      unsigned nof_points,      // number of EC points in 'points' array. Must be either (1) single_msm_size if MSMs are
                                // sharing points or (2) single_msm_size*batch_size otherwise
      P* final_result,
      bool are_scalars_on_device,
      bool are_scalars_montgomery_form,
      bool are_points_on_device,
      bool are_points_montgomery_form,
      bool are_results_on_device,
      bool is_big_triangle,
      int large_bucket_factor,
      int precompute_factor,
      bool is_async,
      cudaStream_t stream)
    {
      CHK_INIT_IF_RETURN();

      const unsigned nof_scalars = batch_size * single_msm_size; // assuming scalars not shared between batch elements
      const bool is_nof_points_valid = ((single_msm_size * batch_size) % nof_points == 0);
      if (!is_nof_points_valid) {
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "bucket_method_msm: #points must be divisible by single_msm_size*batch_size");
      }
      if ((precompute_factor & (precompute_factor - 1)) != 0) {
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument,
          "bucket_method_msm: precompute factors that are not powers of 2 currently unsupported");
      }

      S* d_scalars;
      if (!are_scalars_on_device) {
        // copy scalars to gpu
        CHK_IF_RETURN(cudaMallocAsync(&d_scalars, sizeof(S) * nof_scalars, stream));
        CHK_IF_RETURN(cudaMemcpyAsync(d_scalars, scalars, sizeof(S) * nof_scalars, cudaMemcpyHostToDevice, stream));
      } else {
        d_scalars = scalars;
      }

      if (are_scalars_montgomery_form) {
        if (are_scalars_on_device) {
          S* d_mont_scalars;
          CHK_IF_RETURN(cudaMallocAsync(&d_mont_scalars, sizeof(S) * nof_scalars, stream));
          CHK_IF_RETURN(mont::FromMontgomery(d_scalars, nof_scalars, stream, d_mont_scalars));
          d_scalars = d_mont_scalars;
        } else
          CHK_IF_RETURN(mont::FromMontgomery(d_scalars, nof_scalars, stream, d_scalars));
      }

      unsigned total_bms_per_msm = (bitsize + c - 1) / c;
      unsigned nof_bms_per_msm = (total_bms_per_msm - 1) / precompute_factor + 1;
      unsigned input_indexes_count = nof_scalars * total_bms_per_msm;

      unsigned bm_bitsize = (unsigned)ceil(log2(nof_bms_per_msm));

      unsigned* bucket_indices;
      unsigned* point_indices;
      unsigned* sorted_bucket_indices;
      unsigned* sorted_point_indices;
      CHK_IF_RETURN(cudaMallocAsync(&bucket_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&point_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_indices, sizeof(unsigned) * input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sorted_point_indices, sizeof(unsigned) * input_indexes_count, stream));

      // split scalars into digits
      unsigned NUM_THREADS = 1 << 10;
      unsigned NUM_BLOCKS = (nof_scalars + NUM_THREADS - 1) / NUM_THREADS;

      split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        bucket_indices, point_indices, d_scalars, nof_scalars, nof_points, single_msm_size, total_bms_per_msm,
        bm_bitsize, c, nof_bms_per_msm);

      // sort indices - the indices are sorted from smallest to largest in order to group together the points that
      // belong to each bucket
      unsigned* sort_indices_temp_storage{};
      size_t sort_indices_temp_storage_bytes;
      // The second to last parameter is the default value supplied explicitly to allow passing the stream
      // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for
      // more info
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, input_indexes_count, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
      // The second to last parameter is the default value supplied explicitly to allow passing the stream
      // See https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html#a65e82152de448c6373ed9563aaf8af7e for
      // more info
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
        point_indices, sorted_point_indices, input_indexes_count, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(point_indices, stream));

      // compute number of bucket modules and number of buckets in each module
      unsigned nof_bms_in_batch = nof_bms_per_msm * batch_size;
      // minus nof_bms_per_msm because zero bucket is not included in each bucket module
      const unsigned nof_buckets = (nof_bms_per_msm << c) - nof_bms_per_msm;
      const unsigned total_nof_buckets = nof_buckets * batch_size;

      // find bucket_sizes
      unsigned* single_bucket_indices;
      unsigned* bucket_sizes;
      unsigned* nof_buckets_to_compute;
      // +1 here and in other places because there still is zero index corresponding to zero bucket at this point
      CHK_IF_RETURN(cudaMallocAsync(&single_bucket_indices, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      CHK_IF_RETURN(cudaMallocAsync(&bucket_sizes, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      CHK_IF_RETURN(cudaMallocAsync(&nof_buckets_to_compute, sizeof(unsigned), stream));
      unsigned* encode_temp_storage{};
      size_t encode_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
        nof_buckets_to_compute, input_indexes_count, stream));
      CHK_IF_RETURN(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, sorted_bucket_indices, single_bucket_indices, bucket_sizes,
        nof_buckets_to_compute, input_indexes_count, stream));
      CHK_IF_RETURN(cudaFreeAsync(encode_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_indices, stream));

      // get offsets - where does each new bucket begin
      unsigned* bucket_offsets;
      CHK_IF_RETURN(cudaMallocAsync(&bucket_offsets, sizeof(unsigned) * (total_nof_buckets + 1), stream));
      unsigned* offsets_temp_storage{};
      size_t offsets_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets + 1, stream));
      CHK_IF_RETURN(cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, total_nof_buckets + 1, stream));
      CHK_IF_RETURN(cudaFreeAsync(offsets_temp_storage, stream));

      A* d_points;
      cudaStream_t stream_points;
      if (!are_points_on_device || are_points_montgomery_form) CHK_IF_RETURN(cudaStreamCreate(&stream_points));
      if (!are_points_on_device) {
        // copy points to gpu
        CHK_IF_RETURN(cudaMallocAsync(&d_points, sizeof(A) * nof_points, stream_points));
        CHK_IF_RETURN(cudaMemcpyAsync(d_points, points, sizeof(A) * nof_points, cudaMemcpyHostToDevice, stream_points));
      } else {
        d_points = points;
      }

      if (are_points_montgomery_form) {
        if (are_points_on_device) {
          A* d_mont_points;
          CHK_IF_RETURN(cudaMallocAsync(&d_mont_points, sizeof(A) * nof_points, stream_points));
          CHK_IF_RETURN(mont::FromMontgomery(d_points, nof_points, stream_points, d_mont_points));
          d_points = d_mont_points;
        } else
          CHK_IF_RETURN(mont::FromMontgomery(d_points, nof_points, stream_points, d_points));
      }
      cudaEvent_t event_points_uploaded;
      if (!are_points_on_device || are_points_montgomery_form) {
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_points_uploaded, cudaEventDisableTiming));
        CHK_IF_RETURN(cudaEventRecord(event_points_uploaded, stream_points));
      }

      P* buckets;
      CHK_IF_RETURN(cudaMallocAsync(&buckets, sizeof(P) * (total_nof_buckets + nof_bms_in_batch), stream));

      // launch the bucket initialization kernel with maximum threads
      NUM_THREADS = 1 << 10;
      NUM_BLOCKS = (total_nof_buckets + nof_bms_in_batch + NUM_THREADS - 1) / NUM_THREADS;
      initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, total_nof_buckets + nof_bms_in_batch);

      // removing zero bucket, if it exists
      unsigned smallest_bucket_index;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &smallest_bucket_index, single_bucket_indices, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      // maybe zero bucket is empty after all? in this case zero_bucket_offset is set to 0
      unsigned zero_bucket_offset = (smallest_bucket_index == 0) ? 1 : 0;

      // sort by bucket sizes
      unsigned h_nof_buckets_to_compute;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &h_nof_buckets_to_compute, nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_buckets_to_compute, stream));
      h_nof_buckets_to_compute -= zero_bucket_offset;

      unsigned* sorted_bucket_sizes;
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_sizes, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      unsigned* sorted_bucket_offsets;
      CHK_IF_RETURN(cudaMallocAsync(&sorted_bucket_offsets, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      unsigned* sort_offsets_temp_storage{};
      size_t sort_offsets_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, bucket_offsets + zero_bucket_offset, sorted_bucket_offsets, h_nof_buckets_to_compute, 0,
        sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, bucket_offsets + zero_bucket_offset, sorted_bucket_offsets, h_nof_buckets_to_compute, 0,
        sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_offsets_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_offsets, stream));

      unsigned* sorted_single_bucket_indices;
      CHK_IF_RETURN(
        cudaMallocAsync(&sorted_single_bucket_indices, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
      unsigned* sort_single_temp_storage{};
      size_t sort_single_temp_storage_bytes = 0;
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, single_bucket_indices + zero_bucket_offset, sorted_single_bucket_indices,
        h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaMallocAsync(&sort_single_temp_storage, sort_single_temp_storage_bytes, stream));
      CHK_IF_RETURN(cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, bucket_sizes + zero_bucket_offset,
        sorted_bucket_sizes, single_bucket_indices + zero_bucket_offset, sorted_single_bucket_indices,
        h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream));
      CHK_IF_RETURN(cudaFreeAsync(sort_single_temp_storage, stream));
      CHK_IF_RETURN(cudaFreeAsync(bucket_sizes, stream));
      CHK_IF_RETURN(cudaFreeAsync(single_bucket_indices, stream));

      // find large buckets
      unsigned average_bucket_size = single_msm_size / (1 << c);
      // how large a bucket must be to qualify as a "large bucket"
      unsigned bucket_th = large_bucket_factor * average_bucket_size;
      unsigned* nof_large_buckets;
      CHK_IF_RETURN(cudaMallocAsync(&nof_large_buckets, sizeof(unsigned), stream));
      CHK_IF_RETURN(cudaMemset(nof_large_buckets, 0, sizeof(unsigned)));

      unsigned TOTAL_THREADS = 129000; // TODO: device dependent
      unsigned cutoff_run_length = max(2, h_nof_buckets_to_compute / TOTAL_THREADS);
      unsigned cutoff_nof_runs = (h_nof_buckets_to_compute + cutoff_run_length - 1) / cutoff_run_length;
      NUM_THREADS = 1 << 5;
      NUM_BLOCKS = (cutoff_nof_runs + NUM_THREADS - 1) / NUM_THREADS;
      if (h_nof_buckets_to_compute > 0 && bucket_th > 0)
        find_cutoff_kernel<S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
          sorted_bucket_sizes, h_nof_buckets_to_compute, bucket_th, cutoff_run_length, nof_large_buckets);
      unsigned h_nof_large_buckets;
      CHK_IF_RETURN(
        cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_large_buckets, stream));

      if (!are_points_on_device || are_points_montgomery_form) {
        // by this point, points need to be already uploaded and un-Montgomeried
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_points_uploaded));
        CHK_IF_RETURN(cudaEventDestroy(event_points_uploaded));
        CHK_IF_RETURN(cudaStreamDestroy(stream_points));
      }

      cudaStream_t stream_large_buckets;
      cudaEvent_t event_large_buckets_accumulated;
      // this is where handling of large buckets happens (if there are any)
      if (h_nof_large_buckets > 0 && bucket_th > 0) {
        CHK_IF_RETURN(cudaStreamCreate(&stream_large_buckets));
        CHK_IF_RETURN(cudaEventCreateWithFlags(&event_large_buckets_accumulated, cudaEventDisableTiming));

        unsigned* sorted_bucket_sizes_sum;
        CHK_IF_RETURN(cudaMallocAsync(
          &sorted_bucket_sizes_sum, sizeof(unsigned) * (h_nof_large_buckets + 1), stream_large_buckets));
        CHK_IF_RETURN(cudaMemsetAsync(sorted_bucket_sizes_sum, 0, sizeof(unsigned), stream_large_buckets));
        unsigned* large_bucket_temp_storage{};
        size_t large_bucket_temp_storage_bytes = 0;
        CHK_IF_RETURN(cub::DeviceScan::InclusiveSum(
          large_bucket_temp_storage, large_bucket_temp_storage_bytes, sorted_bucket_sizes, sorted_bucket_sizes_sum + 1,
          h_nof_large_buckets, stream_large_buckets));
        CHK_IF_RETURN(
          cudaMallocAsync(&large_bucket_temp_storage, large_bucket_temp_storage_bytes, stream_large_buckets));
        CHK_IF_RETURN(cub::DeviceScan::InclusiveSum(
          large_bucket_temp_storage, large_bucket_temp_storage_bytes, sorted_bucket_sizes, sorted_bucket_sizes_sum + 1,
          h_nof_large_buckets, stream_large_buckets));
        CHK_IF_RETURN(cudaFreeAsync(large_bucket_temp_storage, stream_large_buckets));
        unsigned h_nof_pts_in_large_buckets;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_nof_pts_in_large_buckets, sorted_bucket_sizes_sum + h_nof_large_buckets, sizeof(unsigned),
          cudaMemcpyDeviceToHost, stream_large_buckets));
        unsigned h_largest_bucket;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_largest_bucket, sorted_bucket_sizes, sizeof(unsigned), cudaMemcpyDeviceToHost, stream_large_buckets));

        // the number of threads for large buckets has an extra h_nof_large_buckets term to account for bucket sizes
        // unevenly divisible by average_bucket_size. there are similar corrections elsewhere when accessing large
        // buckets
        unsigned large_buckets_nof_threads =
          (h_nof_pts_in_large_buckets + average_bucket_size - 1) / average_bucket_size + h_nof_large_buckets;
        unsigned log_nof_large_buckets = (unsigned)ceil(log2(h_nof_large_buckets));
        unsigned* large_bucket_indices;
        CHK_IF_RETURN(cudaMallocAsync(&large_bucket_indices, sizeof(unsigned) * large_buckets_nof_threads, stream));
        NUM_THREADS = min(1 << 8, h_nof_large_buckets);
        NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        initialize_large_bucket_indices<P><<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          sorted_bucket_sizes_sum, average_bucket_size, h_nof_large_buckets, log_nof_large_buckets,
          large_bucket_indices);

        P* large_buckets;
        CHK_IF_RETURN(cudaMallocAsync(&large_buckets, sizeof(P) * large_buckets_nof_threads, stream_large_buckets));

        NUM_THREADS = min(1 << 8, large_buckets_nof_threads);
        NUM_BLOCKS = (large_buckets_nof_threads + NUM_THREADS - 1) / NUM_THREADS;
        accumulate_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          large_buckets, sorted_bucket_offsets, sorted_bucket_sizes, large_bucket_indices, sorted_point_indices,
          d_points, h_nof_large_buckets, c, average_bucket_size, log_nof_large_buckets, large_buckets_nof_threads);

        NUM_THREADS = min(MAX_TH, h_nof_large_buckets);
        NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        // normalization is needed to update buckets sizes and offsets due to reduction that already took place
        normalize_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          sorted_bucket_sizes_sum, average_bucket_size, h_nof_large_buckets);
        // reduce
        for (int s = h_largest_bucket; s > 1; s = ((s + 1) >> 1)) {
          NUM_THREADS = min(MAX_TH, h_nof_large_buckets);
          NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
          normalize_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
            sorted_bucket_sizes, s == h_largest_bucket ? average_bucket_size : 2, h_nof_large_buckets);
          NUM_THREADS = min(MAX_TH, large_buckets_nof_threads);
          NUM_BLOCKS = (large_buckets_nof_threads + NUM_THREADS - 1) / NUM_THREADS;
          sum_reduction_variable_size_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
            large_buckets, sorted_bucket_sizes_sum, sorted_bucket_sizes, large_bucket_indices,
            large_buckets_nof_threads);
        }
        CHK_IF_RETURN(cudaFreeAsync(large_bucket_indices, stream_large_buckets));

        // distribute
        NUM_THREADS = min(MAX_TH, h_nof_large_buckets);
        NUM_BLOCKS = (h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        distribute_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream_large_buckets>>>(
          large_buckets, buckets, sorted_bucket_sizes_sum, sorted_single_bucket_indices, h_nof_large_buckets,
          nof_buckets + nof_bms_per_msm, c + bm_bitsize);
        CHK_IF_RETURN(cudaFreeAsync(large_buckets, stream_large_buckets));
        CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_sizes_sum, stream_large_buckets));

        CHK_IF_RETURN(cudaEventRecord(event_large_buckets_accumulated, stream_large_buckets));
      }

      // launch the accumulation kernel with maximum threads
      if (h_nof_buckets_to_compute > h_nof_large_buckets) {
        NUM_THREADS = 1 << 8;
        NUM_BLOCKS = (h_nof_buckets_to_compute - h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
          buckets, sorted_bucket_offsets + h_nof_large_buckets, sorted_bucket_sizes + h_nof_large_buckets,
          sorted_single_bucket_indices + h_nof_large_buckets, sorted_point_indices, d_points,
          nof_buckets + nof_bms_per_msm, h_nof_buckets_to_compute - h_nof_large_buckets, c + bm_bitsize, c);
      }
      CHK_IF_RETURN(cudaFreeAsync(sorted_point_indices, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_sizes, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_bucket_offsets, stream));
      CHK_IF_RETURN(cudaFreeAsync(sorted_single_bucket_indices, stream));
      if (h_nof_large_buckets > 0 && bucket_th > 0) {
        // all the large buckets need to be accumulated before the final summation
        CHK_IF_RETURN(cudaStreamWaitEvent(stream, event_large_buckets_accumulated));
        CHK_IF_RETURN(cudaStreamDestroy(stream_large_buckets));
      }

      P* d_final_result;
      if (!are_results_on_device) CHK_IF_RETURN(cudaMallocAsync(&d_final_result, sizeof(P) * batch_size, stream));

      unsigned nof_empty_bms_per_batch = 0; // for non-triangle accumluation this may be >0
      P* final_results;
      if (is_big_triangle || c == 1) {
        CHK_IF_RETURN(cudaMallocAsync(&final_results, sizeof(P) * nof_bms_in_batch, stream));
        // launch the bucket module sum kernel - a thread for each bucket module
        NUM_THREADS = 32;
        NUM_BLOCKS = (nof_bms_in_batch + NUM_THREADS - 1) / NUM_THREADS;
        big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, final_results, nof_bms_in_batch, c);
      } else {
        unsigned source_bits_count = c;
        unsigned source_windows_count = nof_bms_per_msm;
        unsigned source_buckets_count = nof_buckets + nof_bms_per_msm;
        unsigned target_windows_count = 0;
        P* source_buckets = buckets;
        buckets = nullptr;
        P* target_buckets;
        P* temp_buckets1;
        P* temp_buckets2;
        for (unsigned i = 0;; i++) {
          const unsigned target_bits_count = (source_bits_count + 1) >> 1;                 // c/2=8
          target_windows_count = source_windows_count << 1;                                // nof bms*2 = 32
          const unsigned target_buckets_count = target_windows_count << target_bits_count; // bms*2^c = 32*2^8
          CHK_IF_RETURN(cudaMallocAsync(
            &target_buckets, sizeof(P) * target_buckets_count * batch_size, stream)); // 32*2^8*2^7 buckets
          CHK_IF_RETURN(cudaMallocAsync(
            &temp_buckets1, sizeof(P) * source_buckets_count / 2 * batch_size, stream)); // 32*2^8*2^7 buckets
          CHK_IF_RETURN(cudaMallocAsync(
            &temp_buckets2, sizeof(P) * source_buckets_count / 2 * batch_size, stream)); // 32*2^8*2^7 buckets

          if (source_bits_count > 0) {
            for (unsigned j = 0; j < target_bits_count; j++) {
              const bool is_first_iter = (j == 0);
              const bool is_last_iter = (j == target_bits_count - 1);
              unsigned nof_threads =
                (((target_buckets_count - target_windows_count) >> 1) << (target_bits_count - 1 - j)) * batch_size;
              NUM_THREADS = min(MAX_TH, nof_threads);
              NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
              single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                is_first_iter ? source_buckets : temp_buckets1, is_last_iter ? target_buckets : temp_buckets1,
                1 << (source_bits_count - j), is_last_iter ? 1 << target_bits_count : 0, 0 /*=write_phase*/,
                (1 << target_bits_count) - 1, nof_threads);

              single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                is_first_iter ? source_buckets : temp_buckets2, is_last_iter ? target_buckets : temp_buckets2,
                1 << (target_bits_count - j), is_last_iter ? 1 << target_bits_count : 0, 1 /*=write_phase*/,
                (1 << target_bits_count) - 1, nof_threads);
            }
          }
          if (target_bits_count == 1) {
            // Note: the reduction ends up with 'target_windows_count' windows per batch element. Some are guaranteed to
            // be empty when target_windows_count>bitsize.
            // for example consider bitsize=253 and c=2. The reduction ends with 254 bms but the most significant one is
            // guaranteed to be zero since the scalars are 253b.
            nof_bms_per_msm = target_windows_count;
            nof_empty_bms_per_batch = target_windows_count > bitsize ? target_windows_count - bitsize : 0;
            nof_bms_in_batch = nof_bms_per_msm * batch_size;

            CHK_IF_RETURN(cudaMallocAsync(&final_results, sizeof(P) * nof_bms_in_batch, stream));
            NUM_THREADS = 32;
            NUM_BLOCKS = (nof_bms_in_batch + NUM_THREADS - 1) / NUM_THREADS;
            last_pass_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(target_buckets, final_results, nof_bms_in_batch);
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
          source_windows_count = target_windows_count;
          source_buckets_count = target_buckets_count;
        }
      }

      // launch the double and add kernel, a single thread per batch element
      NUM_THREADS = 32;
      NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS;
      final_accumulation_kernel<P, S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        final_results, are_results_on_device ? final_result : d_final_result, batch_size, nof_bms_per_msm,
        nof_empty_bms_per_batch, c);
      CHK_IF_RETURN(cudaFreeAsync(final_results, stream));

      if (!are_results_on_device)
        CHK_IF_RETURN(
          cudaMemcpyAsync(final_result, d_final_result, sizeof(P) * batch_size, cudaMemcpyDeviceToHost, stream));

      // free memory
      if (!are_scalars_on_device || are_scalars_montgomery_form) CHK_IF_RETURN(cudaFreeAsync(d_scalars, stream));
      if (!are_points_on_device || are_points_montgomery_form) CHK_IF_RETURN(cudaFreeAsync(d_points, stream));
      if (!are_results_on_device) CHK_IF_RETURN(cudaFreeAsync(d_final_result, stream));
      CHK_IF_RETURN(cudaFreeAsync(buckets, stream));

      if (!is_async) CHK_IF_RETURN(cudaStreamSynchronize(stream));

      return CHK_LAST();
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
    const int bitsize = (config.bitsize == 0) ? S::NBITS : config.bitsize;
    cudaStream_t& stream = config.ctx.stream;

    unsigned c = config.batch_size > 1 ? ((config.c == 0) ? get_optimal_c(msm_size) : config.c) : 16;
    // reduce c to closest power of two (from below) if not using big_triangle reduction logic
    // TODO: support arbitrary values of c
    if (!config.is_big_triangle) {
      while ((c & (c - 1)) != 0)
        c &= (c - 1);
    }

    return CHK_STICKY(bucket_method_msm(
      bitsize, c, scalars, points, config.batch_size, msm_size,
      (config.points_size == 0) ? msm_size : config.points_size, results, config.are_scalars_on_device,
      config.are_scalars_montgomery_form, config.are_points_on_device, config.are_points_montgomery_form,
      config.are_results_on_device, config.is_big_triangle, config.large_bucket_factor, config.precompute_factor,
      config.is_async, stream));
  }

  template <typename A, typename P>
  cudaError_t PrecomputeMSMBases(
    A* bases,
    int bases_size,
    int precompute_factor,
    bool are_bases_on_device,
    device_context::DeviceContext& ctx,
    A* output_bases)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t& stream = ctx.stream;

    CHK_IF_RETURN(cudaMemcpyAsync(
      output_bases, bases, sizeof(A) * bases_size,
      are_bases_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));

    unsigned c = 16;
    unsigned total_nof_bms = (P::SCALAR_FF_NBITS - 1) / c + 1;
    unsigned shift = c * ((total_nof_bms - 1) / precompute_factor + 1);

    unsigned NUM_THREADS = 1 << 8;
    unsigned NUM_BLOCKS = (bases_size + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 1; i < precompute_factor; i++) {
      left_shift_kernel<A, P><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        &output_bases[(i - 1) * bases_size], shift, bases_size, &output_bases[i * bases_size]);
    }

    return CHK_LAST();
  }

  /**
   * Extern "C" version of [PrecomputeMSMBases](@ref PrecomputeMSMBases) function with the following values of template
   * parameters (where the curve is given by `-DCURVE` env variable during build):
   *  - `A` is the [affine representation](@ref affine_t) of curve points;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, PrecomputeMSMBases)(
    curve_config::affine_t* bases,
    int bases_size,
    int precompute_factor,
    bool are_bases_on_device,
    device_context::DeviceContext& ctx,
    curve_config::affine_t* output_bases)
  {
    return PrecomputeMSMBases<curve_config::affine_t, curve_config::projective_t>(
      bases, bases_size, precompute_factor, are_bases_on_device, ctx, output_bases);
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

#if defined(G2_DEFINED)

  /**
   * Extern "C" version of [PrecomputeMSMBases](@ref PrecomputeMSMBases) function with the following values of template
   * parameters (where the curve is given by `-DCURVE` env variable during build):
   *  - `A` is the [affine representation](@ref g2_affine_t) of G2 curve points;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2PrecomputeMSMBases)(
    curve_config::g2_affine_t* bases,
    int bases_size,
    int precompute_factor,
    bool are_bases_on_device,
    device_context::DeviceContext& ctx,
    curve_config::g2_affine_t* output_bases)
  {
    return PrecomputeMSMBases<curve_config::g2_affine_t, curve_config::g2_projective_t>(
      bases, bases_size, precompute_factor, are_bases_on_device, ctx, output_bases);
  }

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

#endif

} // namespace msm