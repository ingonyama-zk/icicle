use crate::field::{ScalarField, ScalarCfg};
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(ScalarField);
} 