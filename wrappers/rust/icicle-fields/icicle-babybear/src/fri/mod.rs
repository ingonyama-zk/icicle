use crate::field::{BabybearExtensionField, BabybearField};
use icicle_core::impl_fri;

impl_fri!("babybear", babybear_fri, BabybearField);
impl_fri!("babybear_extension", babybear_extension_fri, BabybearExtensionField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::field::{BabybearExtensionField, BabybearField};

    impl_fri_tests!(babybear_fri_test, BabybearField, BabybearField);
    impl_fri_tests!(babybear_extension_fri_test, BabybearField, BabybearExtensionField);
}
