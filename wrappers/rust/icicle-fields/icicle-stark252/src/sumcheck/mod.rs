use crate::field::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("stark252", stark252, ScalarField, ScalarCfg);

// Re-export types from the stark252 module
pub use stark252::{SumcheckWrapper, SumcheckProof};

#[cfg(test)]
pub(crate) mod tests {
    use super::stark252::SumcheckWrapper;
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(stark252, ScalarField);
}
