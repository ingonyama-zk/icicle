#[cfg(feature = "bw6-761")]
use crate::curve::{BaseCfg, BaseField};
use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_poseidon;
use icicle_core::poseidon::{Poseidon, PoseidonConfig, PoseidonConstants};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use core::mem::MaybeUninit;

impl_poseidon!("bls12_377", bls12_377, ScalarField, ScalarCfg);

#[cfg(feature = "bw6-761")]
impl_poseidon!("bw6_761", bw6_761, BaseField, BaseCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(ScalarField, 32, "bls12_377", 56);
}
