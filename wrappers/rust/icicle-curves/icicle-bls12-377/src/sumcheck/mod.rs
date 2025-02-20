use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("bls12_377", bls12_377, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(bls12_377, ScalarField);
}
