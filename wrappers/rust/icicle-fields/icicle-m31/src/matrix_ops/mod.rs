use icicle_core::impl_matrix_ops;

impl_matrix_ops!("m31", m31, crate::field::ScalarField);
impl_matrix_ops!("m31_extension", m31_extension, crate::field::ExtensionField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(crate::field::ScalarField);

    // mod extension { // TODO: add m31 extension matrix tests ? ffi bindings missing
    //     use icicle_core::impl_matrix_ops_tests;
    //     impl_matrix_ops_tests!(crate::field::ExtensionField);
    // }
}
