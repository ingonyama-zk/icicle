use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("grumpkin", grumpkin, ScalarField, ScalarCfg);

// Re-export types from the grumpkin module
pub use grumpkin::{SumcheckWrapper, SumcheckProof};

#[cfg(test)]
pub(crate) mod tests {
    use super::grumpkin::SumcheckWrapper;
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(grumpkin, ScalarField);
}
