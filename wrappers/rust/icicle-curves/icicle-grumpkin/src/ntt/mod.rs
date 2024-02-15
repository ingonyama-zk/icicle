use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
// use icicle_core::impl_ntt;
use icicle_core::ntt::{NTTConfig, NTTDir, NTT};
// use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
// use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl NTT<ScalarField> for ScalarCfg {
    fn ntt_unchecked(
        _input: &HostOrDeviceSlice<ScalarField>,
        _dir: NTTDir,
        _cfg: &NTTConfig<ScalarField>,
        _output: &mut HostOrDeviceSlice<ScalarField>,
    ) -> IcicleResult<()> {
        todo!()
    }

    fn initialize_domain(_primitive_root: ScalarField, _ctx: &DeviceContext) -> IcicleResult<()> {
        todo!()
    }

    fn get_default_ntt_config() -> NTTConfig<'static, ScalarField> {
        todo!()
    }
}

// impl_ntt!("grumpkin", grumpkin, ScalarField, ScalarCfg);

// #[cfg(test)]
// pub(crate) mod tests {
//     use crate::curve::ScalarField;
//     use icicle_core::impl_ntt_tests;
//     use icicle_core::ntt::tests::*;
//     use std::sync::OnceLock;

//     impl_ntt_tests!(ScalarField);
// }
