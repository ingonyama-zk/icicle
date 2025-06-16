use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("koalabear", koalabear, ScalarField, ScalarCfg);
impl_sumcheck!("koalabear_extension", koalabear_extension, ExtensionField, ExtensionCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::koalabear::SumcheckWrapper;
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(koalabear, ScalarField);
    mod extension {
        use super::*;

        impl_sumcheck_tests!(koalabear_extension, ExtensionField);
    }
}
