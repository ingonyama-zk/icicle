
#include "cpu_msm.hpp"
#include "icicle/backend/msm_backend.h"

// This file's purpose is just to register the hpp functions to the Icicle API.

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", (cpu_msm_precompute_bases<affine_t, projective_t>));
REGISTER_MSM_BACKEND("CPU", (cpu_msm<affine_t, projective_t>));

#ifdef G2
REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND("CPU", (cpu_msm_precompute_bases<g2_affine_t, g2_projective_t>));
REGISTER_MSM_G2_BACKEND("CPU", (cpu_msm<g2_affine_t, g2_projective_t>));
#endif 