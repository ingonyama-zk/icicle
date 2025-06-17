use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("bls12_377", bls12_377, ScalarField, ScalarCfg);

// Re-export types from the bls12_377 module
pub use bls12_377::{SumcheckWrapper, SumcheckProof};

#[cfg(test)]
pub(crate) mod tests {
    use super::bls12_377::SumcheckWrapper;
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bls12_377, ScalarField);
}
