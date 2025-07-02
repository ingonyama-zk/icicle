use crate::field::{ExtensionField, ScalarField, ExtensionCfg, ScalarCfg};
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("m31", m31, ScalarField, ScalarCfg);
impl_matrix_ops!("m31_extension", m31_extension, ExtensionField, ExtensionCfg);

#[cfg(test)]
mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(ScalarField);

    mod extension {
        use crate::field::ExtensionField;
        use icicle_core::impl_matrix_ops_tests;
        impl_matrix_ops_tests!(ExtensionField);
    }
} 