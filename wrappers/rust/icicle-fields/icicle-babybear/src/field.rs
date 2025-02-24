use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_scalar_field!("babybear", babybear, SCALAR_LIMBS, ScalarField, ScalarCfg);
#[cfg(not(feature = "no_ext_field"))]
impl_scalar_field!(
    "babybear_extension",
    babybear_extension,
    EXTENSION_LIMBS,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no_ext_field"))]
    use super::ExtensionField;
    use super::ScalarField;
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
    #[cfg(not(feature = "no_ext_field"))]
    mod extension {
        use super::*;
        impl_field_tests!(ExtensionField);
    }
    }
