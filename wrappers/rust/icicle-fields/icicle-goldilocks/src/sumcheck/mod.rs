use crate::field::ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("goldilocks", goldilocks, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(goldilocks, ScalarField);
}
