use icicle_cuda_runtime::{device_context::DeviceContext};

use crate::traits::FieldImpl;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PoseidonKernelsConfiguration {
    t: i32,

    number_of_threads: i32,

    hashes_per_block: i32,
    
    singlehash_block_size: i32
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PoseidonConfig<'a> {
    pub ctx: DeviceContext<'a>,

    kernel_cfg: PoseidonKernelsConfiguration,

    are_inputs_on_device: bool,

    are_outputs_on_device: bool,

    input_is_a_state: bool,

    aligned: bool,

    loop_state: bool,

    /// Whether to run Poseidon asyncronously. If set to `true`, Poseidon will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, Poseidon will block the current CPU thread.
    pub is_async: bool,
}

extern "C" {
    #[link_name = "DefaultPoseidonKernelsConfiguration"]
    fn default_poseidon_kernels_configuration_config() -> PoseidonKernelsConfiguration;

    #[link_name = "DefaultPoseidonConfig"]
    fn default_poseidon_config() -> PoseidonConfig<'static>;
}

impl Default for PoseidonKernelsConfiguration {
    fn default() -> Self {
        unsafe { default_poseidon_kernels_configuration_config() }
    }
}

impl<'a> Default for PoseidonConfig<'a> {
    fn default() -> Self {
        unsafe { default_poseidon_config() }
    }
}

pub trait Poseidon<F: FieldImpl> {

}

pub fn initialize_poseidon_constants<F>(primitive_root: F, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>,
{
    <<F as FieldImpl>::Config as NTT<F>>::initialize_domain(primitive_root, ctx)
}