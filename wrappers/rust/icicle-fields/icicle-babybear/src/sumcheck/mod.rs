use crate::field::ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("babybear", babybear, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(babybear, ScalarField);
}
