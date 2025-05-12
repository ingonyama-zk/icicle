use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("grumpkin", grumpkin, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(grumpkin, ScalarField);
}
