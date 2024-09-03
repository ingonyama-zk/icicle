use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::{DeviceSlice, HostOrDeviceSlice};
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_scalar_field!("m31", m31, SCALAR_LIMBS, ScalarField, ScalarCfg);
impl_scalar_field!(
    "m31_extension",
    m31_extension,
    EXTENSION_LIMBS,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
mod tests {
    use super::{ExtensionField, ScalarField};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
    mod extension {
        use super::*;

        impl_field_tests!(ExtensionField);
    }
}
