use crate::curve::Bls12381Curve;
#[cfg(not(feature = "no_g2"))]
use crate::curve::Bls12381G2Curve;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("bls12_381", bls12_381, Bls12381Curve);
#[cfg(not(feature = "no_g2"))]
impl_msm!("bls12_381_g2", bls12_381_g2, Bls12381G2Curve);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12381Curve;
    #[cfg(not(feature = "no_g2"))]
    use crate::curve::Bls12381G2Curve;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(Bls12381Curve);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_msm_tests!(Bls12381G2Curve);
    }
}
