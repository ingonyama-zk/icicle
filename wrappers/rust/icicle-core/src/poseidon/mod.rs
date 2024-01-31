#[doc(hidden)]
pub mod tests;

use icicle_cuda_runtime::{
    device_context::{get_default_device_context, DeviceContext},
    memory::HostOrDeviceSlice,
};

use crate::{error::IcicleResult, traits::FieldImpl};

/// Struct that encodes Poseidon parameters to be passed into the [poseidon_hash_many](poseidon_hash_many) function.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PoseidonConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,

    are_inputs_on_device: bool,

    are_outputs_on_device: bool,

    /// If true, input is considered to be a states vector, holding the preimages
    /// in aligned or not aligned format. Memory under the input pointer will be used for states
    /// If false, fresh states memory will be allocated and input will be copied into it
    pub input_is_a_state: bool,

    /// If true - input should be already aligned for poseidon permutation.
    /// Aligned format: [0, A, B, 0, C, D, ...] (as you might get by using loop_state)
    /// not aligned format: [A, B, 0, C, D, 0, ...] (as you might get from cudaMemcpy2D)
    pub aligned: bool,

    /// If true, hash results will also be copied in the input pointer in aligned format
    pub loop_state: bool,

    /// Whether to run Poseidon asynchronously. If set to `true`, Poseidon will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, Poseidon will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for PoseidonConfig<'a> {
    fn default() -> Self {
        let ctx = get_default_device_context();
        Self {
            ctx,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            input_is_a_state: false,
            aligned: false,
            loop_state: false,
            is_async: false,
        }
    }
}

pub trait Poseidon<F: FieldImpl> {
    fn initialize_constants(arity: u32, ctx: &DeviceContext) -> IcicleResult<()>;
    fn poseidon_unchecked(
        input: &mut HostOrDeviceSlice<F>,
        output: &mut HostOrDeviceSlice<F>,
        number_of_states: u32,
        arity: u32,
        config: &PoseidonConfig,
    ) -> IcicleResult<()>;
}

/// Preloads poseidon constants on the GPU.
/// This function should be called once in a program lifetime before any calls to [poseidon_hash_many](poseidon_hash_many)
pub fn initialize_poseidon_constants<F>(arity: u32, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon<F>,
{
    <<F as FieldImpl>::Config as Poseidon<F>>::initialize_constants(arity, ctx)
}

/// Computes the poseidon hashes for multiple preimages.
///
/// # Arguments
///
/// * `input` - a pointer to the input data. May point to a vector of preimages or a vector of states filled with preimages.
///
/// * `output` - a pointer to the output data. Must be at least of size [number_of_states](number_of_states)
///
/// * `number_of_states` - number of input blocks of size `arity`
///
/// * `arity` - the arity of the hash function (the size of 1 preimage)
///
/// * `config` - config used to specify extra arguments of the Poseidon.
pub fn poseidon_hash_many<F>(
    input: &mut HostOrDeviceSlice<F>,
    output: &mut HostOrDeviceSlice<F>,
    number_of_states: u32,
    arity: u32,
    config: &PoseidonConfig,
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
                #[link_name = concat!($field_prefix, "InitOptimizedPoseidonConstants")]
                pub(crate) fn initialize_poseidon_constants(arity: u32, ctx: &DeviceContext) -> CudaError;

                #[link_name = concat!($field_prefix, "PoseidonHash")]
                pub(crate) fn hash_many(
                    input: *mut $field,
                    output: *mut $field,
                    number_of_states: u32,
                    arity: u32,
                    config: &PoseidonConfig,
                ) -> CudaError;
            }
        }

        impl Poseidon<$field> for $field_config {
            fn initialize_constants(arity: u32, ctx: &DeviceContext) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::initialize_poseidon_constants(arity, ctx).wrap() }
            }

            fn poseidon_unchecked(
                input: &mut HostOrDeviceSlice<$field>,
                output: &mut HostOrDeviceSlice<$field>,
                number_of_states: u32,
                arity: u32,
                config: &PoseidonConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::hash_many(
                        input.as_mut_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        arity,
                        config,
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_poseidon_tests {
    (
      $field:ident
    ) => {
        const SUPPORTED_ARITIES: [u32; 4] = [2, 4, 8, 11];
        static INIT: OnceLock<()> = OnceLock::new();

        #[test]
        fn test_poseidon_hash_many() {
            INIT.get_or_init(move || init_poseidon::<$field>(&SUPPORTED_ARITIES));
            check_poseidon_hash_many::<$field>()
        }
    };
}
