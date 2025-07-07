use icicle_core::{impl_field, impl_generate_random_ffi, impl_montgomery_convertible_ffi};

use icicle_core::bignum::BigNum;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2; // Goldilocks uses 2 limbs for 64-bit field
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_field!(ScalarField, "goldilocks", SCALAR_LIMBS, true, true);
impl_montgomery_convertible_ffi!(ScalarField, "goldilocks_scalar_convert_montgomery");
impl_generate_random_ffi!(ScalarField, "goldilocks_generate_scalars");

impl_field!(ExtensionField, "goldilocks_extension", EXTENSION_LIMBS, true, true);
impl_montgomery_convertible_ffi!(ExtensionField, "goldilocks_extension_scalar_convert_montgomery");
impl_generate_random_ffi!(ExtensionField, "goldilocks_extension_generate_scalars");

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
