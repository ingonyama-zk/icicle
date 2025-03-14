use icicle_core::impl_fri;

use crate::curve::{ScalarCfg, ScalarField};

impl_fri!("bls12_381", bls12_381_fri, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use icicle_core::{
        impl_fri_tests,
        traits::FieldImpl,
    };
    use icicle_hash::keccak::Keccak256;

    use crate::curve::ScalarField;

    impl_fri_tests!(bls12_381_fri_test, ScalarField, ScalarField, Keccak256::new);
}
