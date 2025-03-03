use crate::field::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("m31", m31, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(m31, ScalarField);
}
