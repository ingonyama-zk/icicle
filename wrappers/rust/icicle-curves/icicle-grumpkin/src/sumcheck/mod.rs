use crate::curve::ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("grumpkin", grumpkin, ScalarField);

// Re-export types from the grumpkin module
pub use grumpkin::{SumcheckProof, SumcheckWrapper};

#[cfg(test)]
pub(crate) mod tests {
    use super::grumpkin::SumcheckWrapper;
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(grumpkin, ScalarField);
}
