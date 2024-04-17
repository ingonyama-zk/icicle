use crate::field::ScalarField;
use icicle_core::impl_polynomial_api;

impl_polynomial_api!("babybear", babybear, ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(babybear, ScalarField);
}
