use icicle_core::impl_fri;

use crate::field::{ExtensionField, ScalarField};

impl_fri!("koalabear", koalabear_fri, ScalarField);
impl_fri!("koalabear_extension", koalabear_extension_fri, ExtensionField);

#[cfg(test)]
mod tests {
    mod koalabear_fri_test {
        use crate::field::ScalarField;
        use icicle_core::{impl_fri_test_with_poseidon, impl_fri_tests};
        impl_fri_tests!(ScalarField, ScalarField);
        impl_fri_test_with_poseidon!(ScalarField, ScalarField);
    }
    mod koalabear_extension_fri_test {
        use crate::field::{ExtensionField, ScalarField};
        use icicle_core::impl_fri_tests;
        impl_fri_tests!(ScalarField, ExtensionField);
    }
}
