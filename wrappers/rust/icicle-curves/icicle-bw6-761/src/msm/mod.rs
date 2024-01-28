use crate::curve::CurveCfg;
#[cfg(feature = "g2")]
use crate::curve::G2CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    error::IcicleResult,
    impl_msm,
    msm::{MSMConfig, MSM},
    traits::IcicleResultWrap,
};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_msm!("bw6_761", bw6_761, CurveCfg);
#[cfg(feature = "g2")]
impl_msm!("bw6_761G2", bw6_761_g2, G2CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::CurveCfg;
    #[cfg(feature = "g2")]
    use crate::curve::G2CurveCfg;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(CurveCfg);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        impl_msm_tests!(G2CurveCfg);
    }
}
