use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};
use icicle_core::impl_sumcheck;

impl_sumcheck!("m31", m31, ScalarField, ScalarCfg);
impl_sumcheck!("m31_extension", m31_extension, ExtensionField, ExtensionCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::m31::SumcheckWrapper;
    use crate::field::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(m31, ScalarField);
    mod extension {
        use super::*;

        impl_sumcheck_tests!(m31_extension, ExtensionField);
    }
}
