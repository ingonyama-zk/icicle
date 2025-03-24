use icicle_core::impl_fri;

use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

impl_fri!("babybear", babybear_fri, ScalarField, ScalarCfg);
impl_fri!(
    "babybear_extension",
    babybear_extension_fri,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::field::{ExtensionField, ScalarField};

    impl_fri_tests!(babybear_fri_test, ScalarField, ScalarField);
    impl_fri_tests!(babybear_extension_fri_test, ScalarField, ExtensionField);
}
