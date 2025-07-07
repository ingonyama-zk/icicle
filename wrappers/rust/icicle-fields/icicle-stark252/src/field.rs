use icicle_core::{impl_field, impl_generate_random, impl_montgomery_convertible};

use icicle_core::bignum::BigNum;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 8;

impl_field!(ScalarField, "stark252", SCALAR_LIMBS, true, true);
impl_montgomery_convertible!(ScalarField, "stark252_scalar_convert_montgomery");
impl_generate_random!(ScalarField, "stark252_generate_scalars");

#[cfg(test)]
mod tests {
    use super::ScalarField;
    use icicle_core::impl_field_tests;

    impl_field_tests!(ScalarField);
}
