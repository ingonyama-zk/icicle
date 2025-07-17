use icicle_core::{impl_field, impl_montgomery_convertible};

use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_field!(ScalarField, "babybear", SCALAR_LIMBS);
impl_montgomery_convertible!(ScalarField, "babybear_scalar_convert_montgomery");

impl_field!(ExtensionField, "babybear_extension", EXTENSION_LIMBS);
impl_montgomery_convertible!(ExtensionField, "babybear_extension_scalar_convert_montgomery");

#[cfg(test)]
mod tests {
    use super::{ExtensionField, ScalarField};
    use icicle_core::impl_field_tests;

    impl_field_tests!(ScalarField);
    mod extension {
        use super::*;

        impl_field_tests!(ExtensionField);
    }
}
