use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_sumcheck;

impl_sumcheck!("babybear", babybear, ScalarField);
impl_sumcheck!("babybear_extension", babybear_extension, ExtensionField);

// Re-export types from the babybear module
pub use babybear::{SumcheckProof as ScalarSumcheckProof, SumcheckWrapper as ScalarSumcheckWrapper};
pub use babybear_extension::{SumcheckProof as ExtensionSumcheckProof, SumcheckWrapper as ExtensionSumcheckWrapper};

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
