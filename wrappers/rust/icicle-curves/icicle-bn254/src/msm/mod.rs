use crate::curve::Bn254Curve;
#[cfg(not(feature = "no_g2"))]
use crate::curve::Bn254G2Curve;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("bn254", bn254, Bn254Curve);
#[cfg(not(feature = "no_g2"))]
impl_msm!("bn254_g2", bn254_g2, Bn254G2Curve);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bn254Curve;
    #[cfg(not(feature = "no_g2"))]
    use crate::curve::Bn254G2Curve;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(Bn254Curve);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_msm_tests!(Bn254G2Curve);
    }
}
