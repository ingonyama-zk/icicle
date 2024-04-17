use crate::curve::ScalarField;
use icicle_core::impl_polynomial_api;

impl_polynomial_api!("bls12_381", bls12_381, ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(bls12_381, ScalarField);
}
