use icicle_core::impl_univariate_polynomial_api;

use crate::field::ScalarField;

impl_univariate_polynomial_api!("koalabear", koalabear, ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(koalabear, ScalarField);
}
