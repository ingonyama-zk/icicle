use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::{
    impl_sumcheck_field,
    sumcheck::Sumcheck,
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

impl_sumcheck_field!("bn254", bn254, ScalarField, ScalarCfg);
#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;
    use icicle_core::sumcheck::tests::*;

    impl_sumcheck_tests!(ScalarField);
}
