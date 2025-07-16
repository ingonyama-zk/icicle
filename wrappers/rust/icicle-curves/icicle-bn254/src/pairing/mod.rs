use crate::curve::{G1Projective, G2Projective};
use icicle_core::bignum::BigNum;
use icicle_core::impl_field;
use icicle_core::impl_pairing;
use icicle_core::pairing::Pairing;
use icicle_runtime::IcicleError;


pub(crate) const PAIRING_TARGET_FIELD_LIMBS: usize = 96;

impl_field!(
    PairingTargetField,
    "bn254_pairing_target_field",
    PAIRING_TARGET_FIELD_LIMBS
);
impl_pairing!("bn254", bn254, G1Projective, G2Projective, PairingTargetField);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_pairing_tests;
    use icicle_core::pairing::tests::*;

    use super::{G1Projective, G2Projective, PairingTargetField};

    impl_pairing_tests!(G1Projective, G2Projective, PairingTargetField);
}
