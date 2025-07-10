use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_ring;

impl_program_ring!("m31", m31, ScalarField);
impl_program_ring!("m31_extension", m31_extension, ExtensionField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::impl_program_tests;

    impl_program_tests!(m31, ScalarField);

    mod extension {
        use super::*;
        impl_program_tests!(m31_extension, ExtensionField);
    }
}
