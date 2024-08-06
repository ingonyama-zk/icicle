
#include "cpu_msm.hpp"
#include "icicle/backend/msm_backend.h"

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU", cpu_msm<projective_t>);
// REGISTER_MSM_BACKEND("CPU", cpu_msm_single_thread<proj_test>);

// REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU_REF", cpu_msm_precompute_bases<aff_test>);
// // REGISTER_MSM_BACKEND("CPU_REF", cpu_msm_ref<proj_test>); // TODO revert to yuval's ref when testing batched msm
// REGISTER_MSM_BACKEND("CPU_REF", cpu_msm_single_thread<proj_test>);