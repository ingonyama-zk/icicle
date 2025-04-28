use crate::curve::Bls12_381ScalarField;
use icicle_core::impl_poseidon;

impl_poseidon!("bls12_381", bls12_381, Bls12_381ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_381ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(Bls12_381ScalarField);
}
