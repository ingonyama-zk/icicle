use crate::curve::{G1Projective, G2Projective};
use icicle_core::{
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    memory::{DeviceSlice, HostOrDeviceSlice},
    IcicleError,
};

impl_msm!("bn254", bn254, G1Projective);
#[cfg(feature = "g2")]
impl_msm!("bn254_g2", bn254_g2, G2Projective);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    use crate::curve::G1Projective;

    impl_msm_tests!(G1Projective);

    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        use crate::curve::G2Projective;
        impl_msm_tests!(G2Projective);
    }
}
