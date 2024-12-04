use crate::field::{ScalarCfg, ScalarField};
use icicle_core::impl_poseidon2;

impl_poseidon2!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(ScalarField);
}
