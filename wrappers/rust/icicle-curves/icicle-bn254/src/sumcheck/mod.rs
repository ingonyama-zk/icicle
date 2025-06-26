use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("bn254", bn254, ScalarField, ScalarCfg);

// Re-export types from the bn254 module
pub use bn254::{SumcheckProof, SumcheckWrapper};

#[cfg(test)]
pub(crate) mod tests {
    use super::bn254::SumcheckWrapper;
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bn254, ScalarField);
}
