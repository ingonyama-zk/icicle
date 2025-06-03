use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2; // Goldilocks uses 2 limbs for 64-bit field
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_scalar_field!("goldilocks", goldilocks, SCALAR_LIMBS, ScalarField, ScalarCfg);
impl_scalar_field!(
    "goldilocks_extension",
    goldilocks_extension,
    EXTENSION_LIMBS,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
mod tests {
    use super::{ExtensionField, ScalarField}; // No extension field
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
    mod extension {
        use super::*;

        impl_field_tests!(ExtensionField);
    }
}
