use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("goldilocks", goldilocks, ScalarField, ScalarCfg);
impl_sumcheck!(
    "goldilocks_extension",
    goldilocks_extension,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
pub(crate) mod tests {
    use super::goldilocks::SumcheckWrapper;
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(goldilocks, ScalarField);
    mod extension {
        use super::*;

        impl_sumcheck_tests!(goldilocks_extension, ExtensionField);
    }
}
