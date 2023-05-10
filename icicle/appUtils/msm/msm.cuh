#pragma once
#include <stdexcept>
#include <cuda.h>

#include "../../primitives/projective.cuh"
#include "../../primitives/affine.cuh"
#include "../../curves/curve_config.cuh"

template <typename S, typename P, typename A>
void bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned size, P* final_result, bool on_device);

template <typename S, typename P, typename A>
void batched_bucket_method_msm(unsigned bitsize, unsigned c, S *scalars, A *points, unsigned batch_size, unsigned msm_size, P* final_results, bool on_device);

template <typename S, typename P, typename A>
void batched_large_msm(S* scalars, A* points, unsigned batch_size, unsigned msm_size, P* result, bool on_device);

template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result, bool on_device);

template <typename S, typename P, typename A>
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result, bool on_device);

template <typename A, typename S, typename P>
void reference_msm(S* scalars, A* a_points, unsigned size);
