#pragma once

#include <stdint.h>
#include "icicle/config_extension.h"

// backend specific flags
#define CUDA_MSM_IS_BIG_TRIANGLE                 "is_big_triangle"
#define CUDA_MSM_LARGE_BUCKET_FACTOR             "large_bucket_factor"
#define CUDA_MSM_LARGE_BUCKET_FACTOR_DEFAULT_VAL (10)

static inline bool is_big_triangle(const icicle::ConfigExtension* ext)
{
  return ext && ext->has(CUDA_MSM_IS_BIG_TRIANGLE) ? ext->get<bool>(CUDA_MSM_IS_BIG_TRIANGLE) : false;
}

static inline int get_large_bucket_factor(const icicle::ConfigExtension* ext)
{
  return ext && ext->has(CUDA_MSM_LARGE_BUCKET_FACTOR) ? ext->get<int>(CUDA_MSM_LARGE_BUCKET_FACTOR)
                                                       : CUDA_MSM_LARGE_BUCKET_FACTOR_DEFAULT_VAL;
}
