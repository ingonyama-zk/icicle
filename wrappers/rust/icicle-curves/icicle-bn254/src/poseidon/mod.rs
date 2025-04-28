use crate::curve::Bn254ScalarField;
use icicle_core::impl_poseidon;

impl_poseidon!("bn254", bn254, Bn254ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bn254ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(Bn254ScalarField);
}
