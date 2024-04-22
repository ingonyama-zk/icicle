use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::{
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::HostOrDeviceSlice,
};

use icicle_core::error::IcicleResult;
use icicle_core::traits::IcicleResultWrap;

pub mod tests;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct KeccakConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,

    /// True if inputs are on device and false if they're on host. Default value: false.
    are_inputs_on_device: bool,

    /// If true, output is preserved on device, otherwise on host. Default value: false.
    are_outputs_on_device: bool,

    /// Whether to run the Keccak asynchronously. If set to `true`, the keccak_hash function will be
    /// non-blocking and you'd need to synchronize it explicitly by running
    /// `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, keccak_hash
    /// function will block the current CPU thread.
    is_async: bool,
}

impl<'a> Default for KeccakConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> KeccakConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        KeccakConfig {
            ctx: DeviceContext::default_for_device(device_id),
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
        }
    }
}

extern "C" {
    pub(crate) fn keccak256_cuda(
        input: *const u8,
        input_block_size: i32,
        number_of_blocks: i32,
        output: *mut u8,
        config: KeccakConfig,
    ) -> CudaError;

    pub(crate) fn keccak512_cuda(
        input: *const u8,
        input_block_size: i32,
        number_of_blocks: i32,
        output: *mut u8,
        config: KeccakConfig,
    ) -> CudaError;
}

pub fn keccak256(
    input: &(impl HostOrDeviceSlice<u8> + ?Sized),
    input_block_size: i32,
    number_of_blocks: i32,
    output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: KeccakConfig,
) -> IcicleResult<()> {
    unsafe {
        keccak256_cuda(
            input.as_ptr(),
            input_block_size,
            number_of_blocks,
            output.as_mut_ptr(),
            config,
        )
        .wrap()
    }
}

pub fn keccak512(
    input: &(impl HostOrDeviceSlice<u8> + ?Sized),
    input_block_size: i32,
    number_of_blocks: i32,
    output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: KeccakConfig,
) -> IcicleResult<()> {
    unsafe {
        keccak512_cuda(
            input.as_ptr(),
            input_block_size,
            number_of_blocks,
            output.as_mut_ptr(),
            config,
        )
        .wrap()
    }
}
