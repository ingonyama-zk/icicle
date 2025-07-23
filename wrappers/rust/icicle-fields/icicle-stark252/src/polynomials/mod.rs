use icicle_core::impl_univariate_polynomial_api;

use crate::field::ScalarField;

impl_univariate_polynomial_api!("stark252", stark252, ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(stark252, ScalarField);
}
