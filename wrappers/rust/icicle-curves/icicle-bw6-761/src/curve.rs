#[cfg(feature = "arkworks")]
use ark_bw6_761::{g1::Config as ArkG1Config, Fq};
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_core::field::Field;
use icicle_core::traits::FieldConfig;
use icicle_core::{impl_curve, impl_field};
use icicle_cuda_runtime::device_context::{get_default_device_context, DeviceContext};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use icicle_bls12_377::curve::BaseField as bls12_377BaseField;

pub(crate) const BASE_LIMBS: usize = 12;

impl_field!(BASE_LIMBS, BaseField, BaseCfg, Fq);
pub type ScalarField = bls12_377BaseField;
impl_curve!("bw6_761", CurveCfg, ScalarField, BaseField);

#[cfg(test)]
mod tests {
    use super::{ScalarField, CurveCfg, BASE_LIMBS};
    use icicle_core::curve::Curve;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_core::tests::*;
    use icicle_core::traits::FieldImpl;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
}
