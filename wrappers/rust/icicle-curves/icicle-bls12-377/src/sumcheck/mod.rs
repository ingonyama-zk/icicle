use crate::curve::Bls12_377ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("bls12_377", bls12_377, Bls12_377ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_377ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bls12_377, Bls12_377ScalarField);
}
