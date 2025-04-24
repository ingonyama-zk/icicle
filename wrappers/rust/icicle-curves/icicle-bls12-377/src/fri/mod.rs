#[cfg(feature = "bw6-761")]
use crate::curve::{BaseCfg, BaseField};
use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::impl_fri;

impl_fri!("bls12_377", bls12_377_fri, ScalarField, ScalarCfg);
#[cfg(feature = "bw6-761")]
impl_fri!("bw6_761", bw6_761, BaseField, BaseCfg);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::ScalarField;

    impl_fri_tests!(bls12_377_fri_test, ScalarField, ScalarField);
}
