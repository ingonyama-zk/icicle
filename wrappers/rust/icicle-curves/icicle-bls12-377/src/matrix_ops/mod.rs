
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("bls12_377", bls12_377, crate::curve::ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(crate::curve::ScalarField);
} 