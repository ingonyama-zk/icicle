use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_vec_ops_field;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::vec_ops::{VecOps, VecOpsConfig, BitReverseConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("babybear", babybear, ScalarField, ScalarCfg);
impl_vec_ops_field!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);

// #[repr(C)]
// #[derive(Debug, Clone)]
// pub struct BitReverseConfig<'a> {
//     /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
//     pub ctx: DeviceContext<'a>,

//     /// True if inputs are on device and false if they're on host. Default value: false.
//     pub are_inputs_on_device: bool,

//     /// If true, output is preserved on device, otherwise on host. Default value: false.
//     pub are_outputs_on_device: bool,

//     /// Whether to run the Keccak asynchronously. If set to `true`, the keccak_hash function will be
//     /// non-blocking and you'd need to synchronize it explicitly by running
//     /// `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, keccak_hash
//     /// function will block the current CPU thread.
//     pub is_async: bool,
// }

// impl<'a> Default for BitReverseConfig<'a> {
//     fn default() -> Self {
//         Self::default_for_device(DEFAULT_DEVICE_ID)
//     }
// }

// impl<'a> BitReverseConfig<'a> {
//     pub fn default_for_device(device_id: usize) -> Self {
//         BitReverseConfig {
//             ctx: DeviceContext::default_for_device(device_id),
//             are_inputs_on_device: false,
//             are_outputs_on_device: false,
//             is_async: false,
//         }
//     }
// }

extern "C" {
    pub(crate) fn babybear_bit_reverse_cuda(
        input: *const ScalarField,
        size: u32,
        config: &BitReverseConfig,
        output: *mut ScalarField
    ) -> CudaError;

    pub(crate) fn babybear_bit_reverse_inplace_cuda(
        input: *mut ScalarField,
        size: u32,
        config: &BitReverseConfig
    ) -> CudaError;
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::impl_vec_add_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_add_tests!(ScalarField);

    mod extension {
        use super::*;

        impl_vec_add_tests!(ExtensionField);
    }
}
