use crate::field::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(stark252, ScalarField);
}
