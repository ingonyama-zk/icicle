use crate::curve::Bls12_377ScalarField;
use icicle_core::impl_poseidon2;

impl_poseidon2!("bls12_377", bls12_377, Bls12_377ScalarField);

#[cfg(feature = "bw6-761")]
use crate::curve::Bls12_377BaseField;
#[cfg(feature = "bw6-761")]
impl_poseidon2!("bw6_761", bw6_761, Bls12_377BaseField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_377ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(Bls12_377ScalarField);
}
