use crate::curve::Bn254ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("bn254", bn254, Bn254ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bn254ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bn254, Bn254ScalarField);
}
