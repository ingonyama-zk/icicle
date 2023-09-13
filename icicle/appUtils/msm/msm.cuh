#pragma once
#ifndef MSM_H
#define MSM_H

// template <typename S, typename P, typename A>
// void bucket_method_msm(
//   unsigned bitsize,
//   unsigned c,
//   S* scalars,
//   A* points,
//   unsigned size,
//   P* final_result,
//   bool on_device,
//   bool big_triangle,
//   cudaStream_t stream);

// template <typename S, typename P, typename A>
// void batched_bucket_method_msm(
//   unsigned bitsize,
//   unsigned c,
//   S* scalars,
//   A* points,
//   unsigned batch_size,
//   unsigned msm_size,
//   P* final_results,
//   bool on_device,
//   cudaStream_t stream);

// template <typename S, typename P, typename A>
// void batched_large_msm(
//   S* scalars, A* points, unsigned batch_size, unsigned msm_size, P* result, bool on_device, cudaStream_t stream);

// template <typename S, typename P, typename A>
// void large_msm(
//   S* scalars,
//   A* points,
//   unsigned size,
//   P* result,
//   bool on_device,
//   bool big_triangle,
//   unsigned large_bucket_factor,
//   cudaStream_t stream);

// template <typename S, typename P, typename A>
// void short_msm(S* h_scalars, A* h_points, unsigned size, P* h_final_result, cudaStream_t stream);

// template <typename A, typename S, typename P>
// void reference_msm(S* scalars, A* a_points, unsigned size);

template <typename S, typename P, typename A>
void msm_internal(
  S* scalars,
  unsigned msm_size,
  bool scalars_on_device,
  A* points,
  unsigned points_size,
  unsigned precompute_factor,
  bool points_montgomery_form,
  bool points_on_device,
  P* results,
  unsigned batch_size,
  bool results_on_device,
  unsigned c,
  unsigned bitsize,
  bool big_triangle,
  unsigned large_bucket_factor,
  unsigned device_id,
  cudaStream_t stream);

template <typename S, typename P, typename A>
void msm(S* scalars, A* points, unsigned size, P* results, unsigned device_id, cudaStream_t stream);

#endif