use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_poseidon;

impl_poseidon!("grumpkin", grumpkin, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(ScalarField);
}
