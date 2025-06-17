use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("bls12_381", bls12_381, ScalarField, ScalarCfg);

// Re-export types from the bls12_381 module
pub use bls12_381::{SumcheckWrapper, SumcheckProof};

#[cfg(test)]
pub(crate) mod tests {
    use super::bls12_381::SumcheckWrapper;
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bls12_381, ScalarField);
}
