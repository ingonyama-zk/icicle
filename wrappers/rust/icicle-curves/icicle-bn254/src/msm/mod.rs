use crate::curve::CurveCfg;
use crate::curve::G2CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("bn254", bn254, CurveCfg);
impl_msm!("bn254_g2", bn254_g2, G2CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::CurveCfg;
    use crate::curve::G2CurveCfg;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(CurveCfg);

    mod g2 {
        use super::*;
        impl_msm_tests!(G2CurveCfg);
    }
}
