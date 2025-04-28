use crate::curve::Bn254ScalarField;
use icicle_core::impl_poseidon2;

impl_poseidon2!("bn254", bn254, Bn254ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bn254ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(Bn254ScalarField);
}
