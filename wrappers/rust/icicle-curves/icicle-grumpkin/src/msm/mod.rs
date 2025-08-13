use crate::curve::G1Projective;
use icicle_core::{
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_runtime::{
    memory::{DeviceSlice, HostOrDeviceSlice},
    IcicleError,
};

impl_msm!("grumpkin", grumpkin, G1Projective);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    use crate::curve::G1Projective;

    impl_msm_tests!(G1Projective);
}
