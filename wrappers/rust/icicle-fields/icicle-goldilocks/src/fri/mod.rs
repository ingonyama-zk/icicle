use icicle_core::impl_fri;

use crate::field::{ScalarCfg, ScalarField};

impl_fri!("goldilocks", goldilocks_fri, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    mod goldilocks_fri_test {
        use icicle_core::impl_fri_tests;
        use crate::field::ScalarField;
        impl_fri_tests!(ScalarField, ScalarField);
    }
}
