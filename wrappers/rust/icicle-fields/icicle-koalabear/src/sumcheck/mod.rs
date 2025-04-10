use crate::field::{ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("koalabear", koalabear, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(koalabear, ScalarField);
}
