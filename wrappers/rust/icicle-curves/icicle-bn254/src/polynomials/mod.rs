use crate::curve::Bn254ScalarField;
use icicle_core::impl_univariate_polynomial_api;

impl_univariate_polynomial_api!("bn254", bn254, Bn254ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(bn254, Bn254ScalarField);
}
