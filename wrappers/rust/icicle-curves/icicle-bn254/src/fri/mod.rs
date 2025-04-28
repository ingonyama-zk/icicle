use icicle_core::impl_fri;

use crate::curve::Bn254ScalarField;

impl_fri!("bn254", bn254_fri, Bn254ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::Bn254ScalarField;

    impl_fri_tests!(bn254_fri_test, Bn254ScalarField, Bn254ScalarField);
}
