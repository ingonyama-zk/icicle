#[doc(hidden)]
pub mod tests;

use icicle_cuda_runtime::{
    device_context::{get_default_device_context, DeviceContext},
    memory::HostOrDeviceSlice,
};

use crate::{error::IcicleResult, traits::FieldImpl};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PoseidonConstants<'a, F: FieldImpl> {
    arity: u32,

    partial_rounds: u32,

    full_rounds_half: u32,

    /// These should be pointers to data allocated on device
    round_constants: &'a [F],
    mds_matrix: &'a [F],
    non_sparse_matrix: &'a [F],
    sparse_matrices: &'a [F],

    /// Domain tag is the first element in the Poseidon state.
    /// For the Merkle tree mode it should equal 2^arity - 1
    domain_tag: F,
}

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
    fn create_optimized_constants<'a>(
        arity: u32,
        full_rounds_half: u32,
        partial_rounds: u32,
        constants: &mut [F],
        ctx: &DeviceContext,
    ) -> IcicleResult<PoseidonConstants<'a, F>>;
    fn load_optimized_constants<'a>(arity: u32, ctx: &DeviceContext) -> IcicleResult<PoseidonConstants<'a, F>>;
    fn poseidon_unchecked(
        input: &mut HostOrDeviceSlice<F>,
        output: &mut HostOrDeviceSlice<F>,
        number_of_states: u32,
        arity: u32,
        constants: &PoseidonConstants<F>,
        config: &PoseidonConfig,
    ) -> IcicleResult<()>;
}

/// Loads pre-calculated poseidon constants on the GPU.
pub fn load_optimized_poseidon_constants<'a, F>(
    arity: u32,
    ctx: &DeviceContext,
) -> IcicleResult<PoseidonConstants<'a, F>>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon<F>,
{
    <<F as FieldImpl>::Config as Poseidon<F>>::load_optimized_constants(arity, ctx)
}

/// Creates new instance of poseidon constants on the GPU.
pub fn create_optimized_poseidon_constants<'a, F>(
    arity: u32,
    ctx: &DeviceContext,
    full_rounds_half: u32,
    partial_rounds: u32,
    constants: &mut [F],
) -> IcicleResult<PoseidonConstants<'a, F>>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon<F>,
{
    <<F as FieldImpl>::Config as Poseidon<F>>::create_optimized_constants(
        arity,
        full_rounds_half,
        partial_rounds,
        constants,
        ctx,
    )
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
/// * `constants` - Poseidon constants.
///
/// * `config` - config used to specify extra arguments of the Poseidon.
pub fn poseidon_hash_many<F>(
    input: &mut HostOrDeviceSlice<F>,
    output: &mut HostOrDeviceSlice<F>,
    number_of_states: u32,
    arity: u32,
    constants: &PoseidonConstants<F>,
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

    <<F as FieldImpl>::Config as Poseidon<F>>::poseidon_unchecked(
        input,
        output,
        number_of_states,
        arity,
        constants,
        &local_cfg,
    )
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
            use crate::poseidon::{$field, $field_config, CudaError, DeviceContext, PoseidonConfig, PoseidonConstants};
            extern "C" {
                #[link_name = concat!($field_prefix, "CreateOptimizedPoseidonConstants")]
                pub(crate) fn _create_optimized_constants(
                    arity: u32,
                    full_rounds_half: u32,
                    partial_rounds: u32,
                    constants: *mut $field,
                    ctx: &DeviceContext,
                    poseidon_constants: *mut PoseidonConstants<$field>,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "InitOptimizedPoseidonConstants")]
                pub(crate) fn _load_optimized_constants(
                    arity: u32,
                    ctx: &DeviceContext,
                    constants: *mut PoseidonConstants<$field>,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "PoseidonHash")]
                pub(crate) fn hash_many(
                    input: *mut $field,
                    output: *mut $field,
                    number_of_states: u32,
                    arity: u32,
                    constants: &PoseidonConstants<$field>,
                    config: &PoseidonConfig,
                ) -> CudaError;
            }
        }

        impl Poseidon<$field> for $field_config {
            fn create_optimized_constants<'a>(
                arity: u32,
                full_rounds_half: u32,
                partial_rounds: u32,
                constants: &mut [$field],
                ctx: &DeviceContext,
            ) -> IcicleResult<PoseidonConstants<'a, $field>> {
                unsafe {
                    let mut poseidon_constants = MaybeUninit::<PoseidonConstants<'a, $field>>::uninit();
                    let err = $field_prefix_ident::_create_optimized_constants(
                        arity,
                        full_rounds_half,
                        partial_rounds,
                        constants as *mut _ as *mut $field,
                        ctx,
                        poseidon_constants.as_mut_ptr(),
                    )
                    .wrap();
                    err.and(Ok(poseidon_constants.assume_init()))
                }
            }

            fn load_optimized_constants<'a>(
                arity: u32,
                ctx: &DeviceContext,
            ) -> IcicleResult<PoseidonConstants<'a, $field>> {
                unsafe {
                    let mut constants = MaybeUninit::<PoseidonConstants<'a, $field>>::uninit();
                    let err = $field_prefix_ident::_load_optimized_constants(arity, ctx, constants.as_mut_ptr()).wrap();
                    err.and(Ok(constants.assume_init()))
                }
            }

            fn poseidon_unchecked(
                input: &mut HostOrDeviceSlice<$field>,
                output: &mut HostOrDeviceSlice<$field>,
                number_of_states: u32,
                arity: u32,
                constants: &PoseidonConstants<$field>,
                config: &PoseidonConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::hash_many(
                        input.as_mut_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        arity,
                        constants,
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
      $field:ident,
      $field_bytes:literal,
      $field_prefix:literal,
      $partial_rounds:literal
    ) => {
        #[test]
        fn test_poseidon_hash_many() {
            check_poseidon_hash_many::<$field>()
        }

        #[test]
        fn test_poseidon_custom_config() {
            check_poseidon_custom_config::<$field>($field_bytes, $field_prefix, $partial_rounds)
        }
    };
}
