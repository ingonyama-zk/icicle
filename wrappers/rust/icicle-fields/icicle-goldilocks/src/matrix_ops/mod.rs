use icicle_core::impl_matrix_ops;

impl_matrix_ops!("goldilocks", goldilocks, crate::field::ScalarField);
impl_matrix_ops!(
    "goldilocks_extension",
    goldilocks_extension,
    crate::field::ExtensionField
);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(crate::field::ScalarField);

    // mod extension { // TODO: add goldilocks extension matrix tests ?
    //     use crate::field::ExtensionField;
    //     use icicle_core::impl_matrix_ops_tests;
    //     impl_matrix_ops_tests!(ExtensionField);
    // }
}
