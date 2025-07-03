use icicle_core::impl_matrix_ops;

impl_matrix_ops!("bw6_761", bw6_761, crate::curve::ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(crate::curve::ScalarField);
}
