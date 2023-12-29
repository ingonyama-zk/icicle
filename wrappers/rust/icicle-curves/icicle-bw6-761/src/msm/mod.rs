use crate::curve::{CurveCfg, G1Affine, G1Projective, ScalarField};
use icicle_core::{
    curve::{Affine, CurveConfig, Projective},
    error::*,
    traits::*,
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_cuda_runtime::error::CudaError;

impl_msm!("bw6_761", CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::check_msm;

    use crate::curve::{CurveCfg, ScalarCfg};

    impl_msm_tests!(CurveCfg, ScalarCfg);
}
