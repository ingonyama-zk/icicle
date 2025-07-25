use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_ring;

impl_program_ring!("babybear", babybear, ScalarField);
impl_program_ring!("babybear_extension", babybear_extension, ExtensionField);

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
