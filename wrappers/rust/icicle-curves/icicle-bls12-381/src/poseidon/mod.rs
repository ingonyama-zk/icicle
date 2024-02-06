use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_poseidon;
use icicle_core::poseidon::{Poseidon, PoseidonConfig, PoseidonConstants};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use core::mem::MaybeUninit;

impl_poseidon!("bls12_381", bls12_381, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(ScalarField, 32, "bls12_381", 55);
}
