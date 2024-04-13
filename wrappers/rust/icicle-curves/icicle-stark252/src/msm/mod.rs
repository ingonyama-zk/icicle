use crate::curve::CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    error::IcicleResult,
    impl_msm,
    msm::{MSMConfig, MSM},
    traits::IcicleResultWrap,
};
use icicle_cuda_runtime::{error::CudaError, memory::HostOrDeviceSlice};

impl_msm!("stark252", stark252, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::CurveCfg;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(CurveCfg);
}
