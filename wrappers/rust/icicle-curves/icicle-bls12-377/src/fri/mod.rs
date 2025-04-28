#[cfg(feature = "bw6-761")]
use crate::curve::Bls12_377BaseField;
use crate::curve::Bls12_377ScalarField;

use icicle_core::impl_fri;

impl_fri!("bls12_377", bls12_377_fri, Bls12_377ScalarField);
#[cfg(feature = "bw6-761")]
impl_fri!("bw6_761", bw6_761, Bls12_377BaseField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::Bls12_377ScalarField;

    impl_fri_tests!(bls12_377_fri_test, Bls12_377ScalarField, Bls12_377ScalarField);
}
