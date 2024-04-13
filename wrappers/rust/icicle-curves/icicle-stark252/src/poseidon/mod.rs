use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_poseidon;
use icicle_core::poseidon::{Poseidon, PoseidonConfig, PoseidonConstants};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use core::mem::MaybeUninit;

impl_poseidon!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::poseidon::tests::*;
    use icicle_core::{impl_poseidon_custom_config_test, impl_poseidon_tests};

    impl_poseidon_tests!(ScalarField);
    impl_poseidon_custom_config_test!(ScalarField, 32, "stark252", 56);
}
