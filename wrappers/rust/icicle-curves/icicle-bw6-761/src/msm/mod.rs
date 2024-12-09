use crate::curve::Bw6761Curve;
#[cfg(not(feature = "no_g2"))]
use crate::curve::Bw6761G2Curve;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("bw6_761", bw6_761, Bw6761Curve);
#[cfg(not(feature = "no_g2"))]
impl_msm!("bw6_761_g2", bw6_761_g2, Bw6761G2Curve);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bw6761Curve;
    #[cfg(not(feature = "no_g2"))]
    use crate::curve::Bw6761G2Curve;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(Bw6761Curve);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_msm_tests!(Bw6761G2Curve);
    }
}
