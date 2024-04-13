use icicle_core::curve::{Affine, Curve, Projective};
use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_curve, impl_field, impl_scalar_field};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

pub(crate) const SCALAR_LIMBS: usize = 4;
pub(crate) const BASE_LIMBS: usize = 4;

//TODO: make dummy arkworks config here

impl_scalar_field!("stark252", stark252_sf, SCALAR_LIMBS, ScalarField, ScalarCfg, Fr);
impl_field!(BASE_LIMBS, BaseField, BaseCfg, Fq);
impl_curve!(
    "stark252",
    stark252,
    CurveCfg,
    ScalarField,
    BaseField,
    ArkG1Config,
    G1Affine,
    G1Projective
);

#[cfg(test)]
mod tests {
    use super::{CurveCfg, ScalarField, BASE_LIMBS};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::traits::FieldImpl;
    use icicle_core::{impl_curve_tests, impl_field_tests};

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
}
