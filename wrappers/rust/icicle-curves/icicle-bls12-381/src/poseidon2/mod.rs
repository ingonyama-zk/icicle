use crate::curve::Bls12_381ScalarField;
use icicle_core::impl_poseidon2;

impl_poseidon2!("bls12_381", bls12_381, Bls12_381ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_381ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(Bls12_381ScalarField);
}
