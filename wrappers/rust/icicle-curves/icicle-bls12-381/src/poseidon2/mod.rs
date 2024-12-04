use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_poseidon2;

impl_poseidon2!("bls12_381", bls12_381, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(ScalarField);
}
