use crate::curve::{ScalarField, ScalarCfg};
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("bw6_761", bw6_761, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(ScalarField);
} 