use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("babybear", babybear, ScalarField, ScalarCfg);
impl_sumcheck!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);

// Re-export types from the babybear module
pub use babybear::{SumcheckWrapper as ScalarSumcheckWrapper, SumcheckProof as ScalarSumcheckProof};
pub use babybear_extension::{SumcheckWrapper as ExtensionSumcheckWrapper, SumcheckProof as ExtensionSumcheckProof};

#[cfg(test)]
pub(crate) mod tests {
    use super::babybear::SumcheckWrapper;
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(babybear, ScalarField);
    mod extension {
        use super::*;

        impl_sumcheck_tests!(babybear_extension, ExtensionField);
    }
}
