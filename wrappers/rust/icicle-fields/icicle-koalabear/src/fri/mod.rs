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
    use icicle_core::{
        impl_fri_tests,
        traits::FieldImpl,
    };
    use icicle_hash::keccak::Keccak256;

    use crate::field::{ExtensionField, ScalarField};

    impl_fri_tests!(koalabear_fri_test, ScalarField, ScalarField, Keccak256::new);
    impl_fri_tests!(koalabear_extension_fri_test, ScalarField, ExtensionField, Keccak256::new);
}
