use crate::curve::CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    error::IcicleResult,
    traits::IcicleResultWrap,
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_cuda_runtime::error::CudaError;

impl_msm!("bls12_381", CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    use crate::curve::CurveCfg;

    impl_msm_tests!(CurveCfg);
}
