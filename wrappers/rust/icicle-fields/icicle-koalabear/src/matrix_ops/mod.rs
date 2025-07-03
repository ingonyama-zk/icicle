use icicle_core::impl_matrix_ops;

impl_matrix_ops!("koalabear", koalabear, crate::field::ScalarField);
impl_matrix_ops!("koalabear_extension", koalabear_extension, crate::field::ExtensionField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(crate::field::ScalarField);

    // mod extension { // TODO: add koalabear extension matrix tests ? ffi bindings missing
    //     use icicle_core::impl_matrix_ops_tests;
    //     impl_matrix_ops_tests!(crate::field::ExtensionField);
    // }
}
