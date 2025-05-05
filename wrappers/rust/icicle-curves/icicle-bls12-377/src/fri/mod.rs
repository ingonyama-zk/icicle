#[cfg(feature = "bw6-761")]
use crate::curve::BaseField;
use crate::curve::ScalarField;

use icicle_core::impl_fri;

impl_fri!("bls12_377", bls12_377_fri, ScalarField);
#[cfg(feature = "bw6-761")]
impl_fri!("bw6_761", bw6_761, BaseField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::ScalarField;

    impl_fri_tests!(bls12_377_fri_test, ScalarField, ScalarField);
}
