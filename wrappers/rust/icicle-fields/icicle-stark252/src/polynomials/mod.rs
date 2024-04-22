use crate::field::{ScalarCfg, ScalarField};
use icicle_core::impl_univariate_polynomial_api;

impl_univariate_polynomial_api!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(stark252, ScalarField);
}
