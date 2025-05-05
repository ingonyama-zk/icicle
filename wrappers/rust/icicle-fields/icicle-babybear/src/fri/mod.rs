use crate::field::{ExtensionField, ScalarField};
use icicle_core::impl_fri;

impl_fri!("babybear", babybear_fri, ScalarField);
impl_fri!("babybear_extension", babybear_extension_fri, ExtensionField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::field::{ExtensionField, ScalarField};

    impl_fri_tests!(babybear_fri_test, ScalarField, ScalarField);
    impl_fri_tests!(babybear_extension_fri_test, ScalarField, ExtensionField);
}
