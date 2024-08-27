use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::{DeviceSlice, HostOrDeviceSlice};
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 8;

impl_scalar_field!("stark252", stark252, SCALAR_LIMBS, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use super::ScalarField;
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
}
