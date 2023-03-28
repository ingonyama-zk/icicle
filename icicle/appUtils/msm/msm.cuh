#pragma once
#include <stdexcept>
#include <cuda.h>

#include "../../primitives/projective.cuh"
#include "../../primitives/affine.cuh"
#include "../../curves/curve_config.cuh"

template <typename S, typename P, typename A>
void batched_large_msm(S* scalars, A* points, unsigned batch_size, unsigned msm_size, P* result);

template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result);

template <typename S, typename P, typename A>
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result);
