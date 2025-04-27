use crate::field::{ExtensionField, ScalarField};
use icicle_core::impl_sumcheck;


impl_sumcheck!("koalabear", koalabear, ScalarField);
impl_sumcheck!("koalabear_extension", koalabear_extension, ExtensionField);

// Re-export types from the koalabear module
pub use koalabear::{SumcheckProof as ScalarSumcheckProof, SumcheckWrapper as ScalarSumcheckWrapper};
pub use koalabear_extension::{SumcheckProof as ExtensionSumcheckProof, SumcheckWrapper as ExtensionSumcheckWrapper};

#[cfg(test)]
pub(crate) mod tests {
    use super::koalabear::SumcheckWrapper;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(koalabear, ScalarField);
    mod extension {
        use super::*;

        impl_sumcheck_tests!(koalabear_extension, ExtensionField);
    }
}
