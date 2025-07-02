use crate::curve::{ScalarField, ScalarCfg};
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("bn254", bn254, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(ScalarField);
} 