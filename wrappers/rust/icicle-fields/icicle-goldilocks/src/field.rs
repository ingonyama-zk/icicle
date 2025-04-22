use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, PrimeField, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2; // Goldilocks uses 2 limbs for 64-bit field

impl_scalar_field!("goldilocks", goldilocks, SCALAR_LIMBS, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use super::ScalarField; // No extension field
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
}
