use crate::curve::Bls12377Curve;
#[cfg(not(feature = "no_g2"))]
use crate::curve::Bls12377G2Curve;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("bls12_377", bls12_377, Bls12377Curve);
#[cfg(not(feature = "no_g2"))]
impl_msm!("bls12_377_g2", bls12_377_g2, Bls12377G2Curve);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12377Curve;
    #[cfg(not(feature = "no_g2"))]
    use crate::curve::Bls12377G2Curve;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(Bls12377Curve);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_msm_tests!(Bls12377G2Curve);
    }
}
