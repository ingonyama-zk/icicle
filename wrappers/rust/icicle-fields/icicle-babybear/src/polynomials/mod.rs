use icicle_core::impl_univariate_polynomial_api;

use crate::field::BabybearField;

impl_univariate_polynomial_api!("babybear", babybear, BabybearField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(babybear, BabybearField);
}
