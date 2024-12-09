use crate::curve::GrumpkinCurve;

use icicle_core::{
    curve::{Affine, Curve, Projective},
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("grumpkin", grumpkin, GrumpkinCurve);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::GrumpkinCurve;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    impl_msm_tests!(GrumpkinCurve);
}
