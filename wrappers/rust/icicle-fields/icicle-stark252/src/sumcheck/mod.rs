use crate::field::ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("stark252", stark252, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_sumcheck_tests;

    use crate::field::ScalarField;

    impl_sumcheck_tests!(stark252, ScalarField);
}
