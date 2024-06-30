use std::ffi::c_void;

use icicle_cuda_runtime::{
    device::check_device,
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::{DeviceSlice, HostOrDeviceSlice},
};

use crate::ntt::IcicleResult;

/// Struct that encodes Sponge hash parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SpongeConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    pub(crate) are_inputs_on_device: bool,
    pub(crate) are_outputs_on_device: bool,
    pub input_rate: u32,
    pub output_rate: u32,
    pub offset: u32,

    /// If true - input should be already aligned for poseidon permutation.
    /// Aligned format: [0, A, B, 0, C, D, ...] (as you might get by using loop_state)
    /// not aligned format: [A, B, 0, C, D, 0, ...] (as you might get from cudaMemcpy2D)
    pub recursive_squeeze: bool,

    /// If true, hash results will also be copied in the input pointer in aligned format
    pub aligned: bool,
    /// Whether to run the sponge operations asynchronously. If set to `true`, the functions will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the functions will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for SpongeConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> SpongeConfig<'a> {
    pub(crate) fn default_for_device(device_id: usize) -> Self {
        SpongeConfig {
            ctx: DeviceContext::default_for_device(device_id),
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            input_rate: 0,
            output_rate: 0,
            offset: 0,
            recursive_squeeze: false,
            aligned: false,
            is_async: false,
        }
    }
}

pub trait SpongeHash<PreImage, Image> {
    fn absorb_many(
        &self,
        inputs: &(impl HostOrDeviceSlice<PreImage> + ?Sized),
        states: &mut DeviceSlice<Image>,
        number_of_states: usize,
        input_block_len: usize,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn squeeze_many(
        &self,
        states: &DeviceSlice<Image>,
        output: &mut (impl HostOrDeviceSlice<Image> + ?Sized),
        number_of_states: usize,
        output_len: usize,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn hash_many(
        &self,
        inputs: &(impl HostOrDeviceSlice<PreImage> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<Image> + ?Sized),
        number_of_states: usize,
        input_block_len: usize,
        output_len: usize,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn default_config<'a>(&self) -> SpongeConfig<'a>;

    fn get_handle(&self) -> *const c_void;
}

pub(crate) fn sponge_check_input<T>(
    inputs: &(impl HostOrDeviceSlice<T> + ?Sized),
    number_of_states: usize,
    input_block_len: usize,
    input_rate: usize,
    ctx: &DeviceContext,
) {
    if input_block_len > input_rate {
        panic!(
            "input block len ({}) can't be greater than input rate ({})",
            input_block_len, input_rate
        );
    }

    let inputs_size_expected = input_block_len * number_of_states;
    if inputs.len() < inputs_size_expected {
        panic!(
            "inputs len is {}; but needs to be at least {}",
            inputs.len(),
            inputs_size_expected,
        );
    }

    let ctx_device_id = ctx.device_id;
    if let Some(device_id) = inputs.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in inputs and context are different"
        );
    }
    check_device(ctx_device_id);
}

pub(crate) fn sponge_check_states<T>(
    states: &DeviceSlice<T>,
    number_of_states: usize,
    width: usize,
    ctx: &DeviceContext,
) {
    let states_size_expected = width * number_of_states;
    if states.len() < states_size_expected {
        panic!(
            "states len is {}; but needs to be at least {}",
            states.len(),
            states_size_expected,
        );
    }

    let ctx_device_id = ctx.device_id;
    if let Some(device_id) = states.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in states and context are different"
        );
    }
    check_device(ctx_device_id);
}

pub(crate) fn sponge_check_outputs<T>(
    outputs: &(impl HostOrDeviceSlice<T> + ?Sized),
    number_of_states: usize,
    output_len: usize,
    width: usize,
    recursive: bool,
    ctx: &DeviceContext,
) {
    let outputs_size_expected = if recursive {
        width * number_of_states
    } else {
        output_len * number_of_states
    };

    if outputs.len() < outputs_size_expected {
        panic!(
            "outputs len is {}; but needs to be at least {}",
            outputs.len(),
            outputs_size_expected,
        );
    }

    let ctx_device_id = ctx.device_id;
    if let Some(device_id) = outputs.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in outputs and context are different"
        );
    }
    check_device(ctx_device_id);
}
