use icicle_core::impl_fri;

use crate::curve::Bls12_381ScalarField;

impl_fri!("bls12_381", bls12_381_fri, Bls12_381ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::Bls12_381ScalarField;

    impl_fri_tests!(bls12_381_fri_test, Bls12_381ScalarField, Bls12_381ScalarField);
}
