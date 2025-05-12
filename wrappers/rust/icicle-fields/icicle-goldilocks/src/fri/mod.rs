use icicle_core::impl_fri;

use crate::field::{ScalarCfg, ScalarField};

impl_fri!("goldilocks", goldilocks_fri, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    mod goldilocks_fri_test {
        use crate::field::ScalarField;
        use icicle_core::impl_fri_tests;
        impl_fri_tests!(ScalarField, ScalarField);
    }
}
