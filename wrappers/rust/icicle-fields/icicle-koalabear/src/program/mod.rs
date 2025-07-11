use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_ring;

impl_program_ring!("koalabear", koalabear, ScalarField);
impl_program_ring!("koalabear_extension", koalabear_extension, ExtensionField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::impl_program_tests;

    impl_program_tests!(koalabear, ScalarField);

    mod extension {
        use super::*;
        impl_program_tests!(koalabear_extension, ExtensionField);
    }
}
