use crate::curve::{Bls12_377BaseField, Bls12_377ScalarField};
use icicle_core::impl_poseidon;

impl_poseidon!("bls12_377", bls12_377, Bls12_377ScalarField);

#[cfg(feature = "bw6-761")]
impl_poseidon!("bw6_761", bw6_761, Bls12_377BaseField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_377ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(Bls12_377ScalarField);
}
