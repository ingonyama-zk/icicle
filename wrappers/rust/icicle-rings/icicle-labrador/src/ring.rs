use icicle_core::field::{Field, MontgomeryConvertibleField};
use icicle_core::traits::{FieldConfig, FieldImpl, GenerateRandom};
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2;

impl_scalar_field!("labrador", labrador, SCALAR_LIMBS, ScalarRing, ScalarCfg);
impl_scalar_field!("labrador_rns", labrador_rns, SCALAR_LIMBS, ScalarRingRns, ScalarCfgRns);

#[cfg(test)]
mod tests {
    use super::{ScalarRing, ScalarRingRns};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarRing);
    mod rns {
        use super::*;
        impl_field_tests!(ScalarRingRns);
    }
}
