use crate::field::{ExtensionField, ScalarField, ExtensionCfg, ScalarCfg};
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("babybear", babybear, ScalarField, ScalarCfg);
impl_matrix_ops!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);

#[cfg(test)]
mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_matrix_ops_tests;
    impl_matrix_ops_tests!(ScalarField);

    // mod extension { // TODO: add babybear extension matrix tests ?
    //     use crate::field::ExtensionField;
    //     use icicle_core::impl_matrix_ops_tests;
    //     impl_matrix_ops_tests!(ExtensionField);
    // }
} 