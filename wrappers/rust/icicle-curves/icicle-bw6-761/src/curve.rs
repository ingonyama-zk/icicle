#[cfg(all(feature = "arkworks", feature = "g2"))]
use ark_bw6_761::g2::Config as ArkG2Config;
#[cfg(feature = "arkworks")]
use ark_bw6_761::{g1::Config as ArkG1Config, Fq};
use icicle_bls12_377::curve::BaseField as bls12_377BaseField;
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_core::field::Field;
use icicle_core::traits::FieldConfig;
use icicle_core::{impl_curve, impl_field};
use icicle_cuda_runtime::device_context::{get_default_device_context, DeviceContext};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

pub(crate) const BASE_LIMBS: usize = 12;

impl_field!(BASE_LIMBS, BaseField, BaseCfg, Fq);
pub type ScalarField = bls12_377BaseField;
impl_curve!(
    "bw6_761",
    bw6_761,
    CurveCfg,
    ScalarField,
    BaseField,
    ArkG1Config,
    G1Affine,
    G1Projective
);
#[cfg(feature = "g2")]
impl_curve!(
    "bw6_761G2",
    bw6_761_g2,
    G2CurveCfg,
    ScalarField,
    BaseField,
    ArkG2Config,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    #[cfg(feature = "g2")]
    use super::G2CurveCfg;
    use super::{CurveCfg, ScalarField, BASE_LIMBS};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::traits::FieldImpl;
    use icicle_core::{impl_curve_tests, impl_field_tests};

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        impl_curve_tests!(BASE_LIMBS, G2CurveCfg);
    }
}
