use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_poseidon;
use icicle_core::poseidon::{Poseidon, PoseidonConfig};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_poseidon!("bls12_381", bls12_381, ScalarField, ScalarCfg);
