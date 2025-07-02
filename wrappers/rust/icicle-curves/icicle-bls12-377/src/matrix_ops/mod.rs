use crate::curve::{ScalarField, ScalarCfg};
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("bls12_377", bls12_377, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(ScalarField);
} 