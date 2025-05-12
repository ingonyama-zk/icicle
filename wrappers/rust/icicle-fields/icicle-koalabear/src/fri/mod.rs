use icicle_core::impl_fri;

use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

impl_fri!("koalabear", koalabear_fri, ScalarField, ScalarCfg);
impl_fri!(
    "koalabear_extension",
    koalabear_extension_fri,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
mod tests {
    mod koalabear_fri_test {
        use icicle_core::{impl_fri_test_with_poseidon, impl_fri_tests};
        use crate::field::ScalarField;
        impl_fri_tests!(ScalarField, ScalarField);
        impl_fri_test_with_poseidon!(ScalarField, ScalarField);
    }
    mod koalabear_extension_fri_test {
        use icicle_core::impl_fri_tests;
        use crate::field::{ExtensionField, ScalarField};
        impl_fri_tests!(ScalarField, ExtensionField);
    }
}
