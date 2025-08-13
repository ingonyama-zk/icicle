use icicle_core::{impl_field, impl_montgomery_convertible};

use icicle_core::bignum::BigNum;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2; // Goldilocks uses 2 limbs for 64-bit field
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_field!(ScalarField, "goldilocks", SCALAR_LIMBS);
impl_montgomery_convertible!(ScalarField, "goldilocks_scalar_convert_montgomery");

impl_field!(ExtensionField, "goldilocks_extension", EXTENSION_LIMBS);
impl_montgomery_convertible!(ExtensionField, "goldilocks_extension_scalar_convert_montgomery");

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
