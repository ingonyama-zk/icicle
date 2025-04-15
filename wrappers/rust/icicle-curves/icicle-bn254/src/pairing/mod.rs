use crate::curve::CurveCfg;
use crate::curve::G2CurveCfg;
use icicle_core::curve::Affine;
use icicle_core::field::Field;
use icicle_core::impl_field;
use icicle_core::impl_pairing;
use icicle_core::pairing::Pairing;
use icicle_core::traits::{FieldConfig, FieldImpl};
use icicle_runtime::eIcicleError;

pub(crate) const PAIRING_TARGET_FIELD_LIMBS: usize = 96;

impl_field!(
    "bn254_pairing_target_field",
    PAIRING_TARGET_FIELD_LIMBS,
    PairingTargetField,
    PairingTargetFieldCfg
);
impl_pairing!("bn254", bn254, CurveCfg, G2CurveCfg, PairingTargetField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::CurveCfg;
    use crate::curve::G2CurveCfg;
    use icicle_core::impl_pairing_tests;
    use icicle_core::pairing::tests::*;

    use super::PairingTargetField;

    impl_pairing_tests!(CurveCfg, G2CurveCfg, PairingTargetField);
}
