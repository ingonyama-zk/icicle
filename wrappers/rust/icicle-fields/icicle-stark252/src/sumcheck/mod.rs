use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::stark252::SumcheckWrapper;
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(stark252, ScalarField);
    mod extension {
        use super::*;

        impl_sumcheck_tests!(stark252_extension, ExtensionField);
    }
}
