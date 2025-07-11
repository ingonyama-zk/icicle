use icicle_core::impl_univariate_polynomial_api;

use crate::field::ScalarField;

impl_univariate_polynomial_api!("babybear", babybear, ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(babybear, ScalarField);
}
