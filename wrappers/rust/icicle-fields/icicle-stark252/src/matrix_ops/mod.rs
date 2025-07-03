use icicle_core::impl_matrix_ops;

impl_matrix_ops!("stark252", stark252, crate::field::ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(crate::field::ScalarField);
} 