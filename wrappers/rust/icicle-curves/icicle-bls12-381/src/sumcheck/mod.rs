use crate::curve::Bls12_381ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("bls12_381", bls12_381, Bls12_381ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_381ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bls12_381, Bls12_381ScalarField);
}
