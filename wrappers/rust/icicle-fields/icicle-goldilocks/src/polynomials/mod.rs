use icicle_core::impl_univariate_polynomial_api;

use crate::field::GoldilocksField;

impl_univariate_polynomial_api!("goldilocks", goldilocks, GoldilocksField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(goldilocks, GoldilocksField);
}
