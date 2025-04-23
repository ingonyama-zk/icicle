use crate::field::{BabybearExtensionField, BabybearField};

use icicle_core::impl_program_field;

impl_program_field!("babybear", babybear, BabybearField);
impl_program_field!("babybear_extension", babybear_extension, BabybearExtensionField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::impl_program_tests;

    impl_program_tests!(babybear, ScalarField);

    mod extension {
        use super::*;
        impl_program_tests!(babybear_extension, ExtensionField);
    }
}
