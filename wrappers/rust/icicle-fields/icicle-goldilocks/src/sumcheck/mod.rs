use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("goldilocks", goldilocks, ScalarField, ScalarCfg);
impl_sumcheck!(
    "goldilocks_extension",
    goldilocks_extension,
    ExtensionField,
    ExtensionCfg
);

// Re-export types from the goldilocks module
pub use goldilocks::{SumcheckWrapper as ScalarSumcheckWrapper, SumcheckProof as ScalarSumcheckProof};
pub use goldilocks_extension::{SumcheckWrapper as ExtensionSumcheckWrapper, SumcheckProof as ExtensionSumcheckProof};

#[cfg(test)]
pub(crate) mod tests {
    use super::goldilocks::SumcheckWrapper;
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(goldilocks, ScalarField);
    mod extension {
        use super::*;

        impl_sumcheck_tests!(goldilocks_extension, ExtensionField);
    }
}
