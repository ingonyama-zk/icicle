use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::{DeviceSlice, HostOrDeviceSlice};
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const COMPLEX_EXTENSION_LIMBS: usize = 2;
pub(crate) const QUARTIC_EXTENSION_LIMBS: usize = 4;

impl_scalar_field!("m31", m31, SCALAR_LIMBS, ScalarField, ScalarCfg);
impl_scalar_field!(
    "m31_complex_extension",
    m31_complex_extension,
    COMPLEX_EXTENSION_LIMBS,
    ComplexExtensionField,
    ComplexExtensionCfg
);
impl_scalar_field!(
    "m31_quartic_extension",
    m31_quartic_extension,
    QUARTIC_EXTENSION_LIMBS,
    QuarticExtensionField,
    QuarticExtensionCfg
);

#[cfg(test)]
mod tests {
    use super::{ComplexExtensionField, QuarticExtensionField, ScalarField};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
    mod complex_extension {
        use super::*;

        impl_field_tests!(ComplexExtensionField);
    }

    mod quartic_extension {
        use super::*;

        impl_field_tests!(QuarticExtensionField);
    }
}
