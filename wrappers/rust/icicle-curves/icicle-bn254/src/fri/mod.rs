use icicle_core::impl_fri;

use crate::curve::{ScalarCfg, ScalarField};

impl_fri!("bn254", bn254_fri, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    mod bn254_fri_test {
        use icicle_core::{impl_fri_test_with_poseidon, impl_fri_tests};
        use crate::curve::ScalarField;
        impl_fri_tests!(ScalarField, ScalarField);
        impl_fri_test_with_poseidon!(ScalarField, ScalarField);
    }
}
