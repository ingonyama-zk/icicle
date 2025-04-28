use crate::curve::Bls12_381ScalarField;
use icicle_core::impl_univariate_polynomial_api;

impl_univariate_polynomial_api!("bls12_381", bls12_381, Bls12_381ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(bls12_381, Bls12_381ScalarField);
}
