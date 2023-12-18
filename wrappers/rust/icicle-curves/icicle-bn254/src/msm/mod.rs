use crate::curve::{CurveCfg, G1Affine, G1Projective, ScalarField};
use icicle_core::{
    curve::{Affine, CurveConfig, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

impl_msm!("bn254", CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::check_msm;

    use crate::curve::{CurveCfg, ScalarCfg};

    impl_msm_tests!(CurveCfg, ScalarCfg);
}
