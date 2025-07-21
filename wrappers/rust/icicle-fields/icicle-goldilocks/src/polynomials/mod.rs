use icicle_core::impl_univariate_polynomial_api;

use crate::field::ScalarField;

impl_univariate_polynomial_api!("goldilocks", goldilocks, ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(goldilocks, ScalarField);
}
