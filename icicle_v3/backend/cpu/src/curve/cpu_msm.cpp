
#include "cpu_msm.hpp"
#include "icicle/backend/msm_backend.h"

// This file's purpose is just to register the hpp functions to the Icicle API.

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU", cpu_msm<projective_t>);