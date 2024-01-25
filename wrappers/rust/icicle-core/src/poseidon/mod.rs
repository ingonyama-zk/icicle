use icicle_cuda_runtime::{device_context::{get_default_device_context, DeviceContext}, memory::HostOrDeviceSlice};

use crate::{error::IcicleResult, traits::FieldImpl};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PoseidonConfig<'a> {
    pub ctx: DeviceContext<'a>,

    are_inputs_on_device: bool,

    are_outputs_on_device: bool,

    pub input_is_a_state: bool,

    pub aligned: bool,

    pub loop_state: bool,

    /// Whether to run Poseidon asyncronously. If set to `true`, Poseidon will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, Poseidon will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for PoseidonConfig<'a> {
    fn default() -> Self {
        let ctx = get_default_device_context();
        PoseidonConfig {
            ctx,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            input_is_a_state: false,
            aligned: false,
            loop_state: false,
            is_async: false
        }
    }
}

pub trait Poseidon<F: FieldImpl> {
    fn initialize_constants(arity: i32, ctx: &DeviceContext) -> IcicleResult<()>;
    fn poseidon_unchecked(input: &mut HostOrDeviceSlice<F>, output: &mut HostOrDeviceSlice<F>, number_of_states: i32, arity: i32, config: &PoseidonConfig) -> IcicleResult<()>;
}

pub fn initialize_poseidon_constants<F>(arity: i32, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon<F>,
{
    <<F as FieldImpl>::Config as Poseidon<F>>::initialize_constants(arity, ctx)
}

pub fn poseidon_hash_many<F>(
    input: &mut HostOrDeviceSlice<F>,
    output: &mut HostOrDeviceSlice<F>,
    number_of_states: i32,
    arity: i32,
    config: &PoseidonConfig
) -> IcicleResult<()> 
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let input_len_required = if config.input_is_a_state {
        number_of_states * (arity + 1)
    } else {
        number_of_states * arity
    };
    
    if input.len() < input_len_required as usize {
        panic!(
            "input len is {}; but needs to be at least {}",
            input.len(),
            input_len_required
        );
    }

    if output.len() < number_of_states as usize {
        panic!(
            "output len is {}; but needs to be at least {}",
            output.len(),
            number_of_states
        );
    }

    let mut local_cfg = config.clone();
    local_cfg.are_inputs_on_device = input.is_on_device();
    local_cfg.are_outputs_on_device = output.is_on_device();

    <<F as FieldImpl>::Config as Poseidon<F>>::poseidon_unchecked(input, output, number_of_states, arity, &local_cfg)
}

#[macro_export]
macro_rules! impl_poseidon {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::poseidon::{$field, $field_config, CudaError, DeviceContext, PoseidonConfig};
            extern "C" {
                #[link_name = concat!($field_prefix, "InitPoseidonConstants")]
                pub(crate) fn initialize_poseidon_constants(arity: i32, ctx: &DeviceContext) -> CudaError;

                #[link_name = concat!($field_prefix, "PoseidonHash")]
                pub(crate) fn hash_many(input: *mut $field, output: *mut $field, number_of_states: i32, arity: i32, config: &PoseidonConfig) -> CudaError;
            }
        }

        impl Poseidon<$field> for $field_config {
            fn initialize_constants(arity: i32, ctx: &DeviceContext) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::initialize_poseidon_constants(arity, ctx).wrap() }
            }

            fn poseidon_unchecked(
                input: &mut HostOrDeviceSlice<$field>,
                output: &mut HostOrDeviceSlice<$field>,
                number_of_states: i32,
                arity: i32,
                config: &PoseidonConfig
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::hash_many(
                        input.as_mut_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        arity,
                        config,
                    ).wrap()
                }
            }
        }
    };
}