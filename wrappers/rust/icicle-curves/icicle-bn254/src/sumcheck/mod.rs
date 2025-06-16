use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("bn254", bn254, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::bn254::SumcheckWrapper;
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bn254, ScalarField);
}
