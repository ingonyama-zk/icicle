use crate::field::{ExtensionField, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("goldilocks", goldilocks, ScalarField);
impl_sumcheck!("goldilocks_extension", goldilocks_extension, ExtensionField);

// Re-export types from the goldilocks module
pub use goldilocks::{SumcheckProof as ScalarSumcheckProof, SumcheckWrapper as ScalarSumcheckWrapper};
pub use goldilocks_extension::{SumcheckProof as ExtensionSumcheckProof, SumcheckWrapper as ExtensionSumcheckWrapper};

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
