use crate::curve::Bls12_377ScalarField;
use icicle_core::impl_univariate_polynomial_api;

impl_univariate_polynomial_api!("bls12_377", bls12_377, Bls12_377ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(bls12_377, Bls12_377ScalarField);
}
