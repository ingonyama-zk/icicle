#[doc(hidden)]
pub mod tests;

use icicle_cuda_runtime::{
    device::check_device,
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::{DeviceSlice, HostOrDeviceSlice},
};

use crate::{error::IcicleResult, traits::FieldImpl};

#[repr(C)]
#[derive(Debug, Clone)]
pub enum MdsType {
    Default,
    Plonky,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub enum PoseidonMode {
    Compression,
    Permutation,
}

#[repr(C)]
pub struct Poseidon2Constants<'a, F: FieldImpl> {
    width: u32,

    alpha: u32,

    internal_rounds: u32,

    external_rounds: u32,

    round_constants: &'a DeviceSlice<F>,

    inernal_matrix_diag: &'a DeviceSlice<F>,
}

impl<F: FieldImpl> std::fmt::Debug for Poseidon2Constants<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {}, {}, {}",
            self.width, self.alpha, self.internal_rounds, self.external_rounds
        )
    }
}

/// Struct that encodes Poseidon parameters to be passed into the [poseidon_hash_many](poseidon_hash_many) function.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Poseidon2Config<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,

    are_states_on_device: bool,

    are_outputs_on_device: bool,

    pub mode: PoseidonMode,

    pub mds_type: MdsType,

    pub output_index: u32,

    /// If true, hash results will also be copied in the input pointer in aligned format
    pub loop_state: bool,

    /// Whether to run Poseidon asynchronously. If set to `true`, Poseidon will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, Poseidon will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for Poseidon2Config<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> Poseidon2Config<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        Self {
            ctx: DeviceContext::default_for_device(device_id),
            are_states_on_device: false,
            are_outputs_on_device: false,
            mode: PoseidonMode::Compression,
            mds_type: MdsType::Default,
            output_index: 1,
            loop_state: false,
            is_async: false,
        }
    }
}

pub trait Poseidon2<F: FieldImpl> {
    fn create_optimized_constants<'a>(
        width: u32,
        alpha: u32,
        external_rounds: u32,
        internal_rounds: u32,
        round_constants: &mut [F],
        internal_matrix_diag: &mut [F],
        ctx: &DeviceContext,
    ) -> IcicleResult<Poseidon2Constants<'a, F>>;
    fn load_optimized_constants<'a>(width: u32, ctx: &DeviceContext) -> IcicleResult<Poseidon2Constants<'a, F>>;
    fn poseidon_unchecked(
        states: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        width: u32,
        constants: &Poseidon2Constants<F>,
        config: &Poseidon2Config,
    ) -> IcicleResult<()>;
}

/// Loads pre-calculated poseidon constants on the GPU.
pub fn load_optimized_poseidon2_constants<'a, F>(
    width: u32,
    ctx: &DeviceContext,
) -> IcicleResult<Poseidon2Constants<'a, F>>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    <<F as FieldImpl>::Config as Poseidon2<F>>::load_optimized_constants(width, ctx)
}

/// Creates new instance of poseidon constants on the GPU.
pub fn create_optimized_poseidon2_constants<'a, F>(
    width: u32,
    alpha: u32,
    ctx: &DeviceContext,
    external_rounds: u32,
    internal_rounds: u32,
    round_constants: &mut [F],
    internal_matrix_diag: &mut [F],
) -> IcicleResult<Poseidon2Constants<'a, F>>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    <<F as FieldImpl>::Config as Poseidon2<F>>::create_optimized_constants(
        width,
        alpha,
        external_rounds,
        internal_rounds,
        round_constants,
        internal_matrix_diag,
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
    states: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    number_of_states: u32,
    width: u32,
    constants: &Poseidon2Constants<F>,
    config: &Poseidon2Config,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    if states.len() < (number_of_states * width) as usize {
        panic!(
            "input len is {}; but needs to be at least {}",
            states.len(),
            number_of_states * width
        );
    }

    if output.len() < number_of_states as usize {
        panic!(
            "output len is {}; but needs to be at least {}",
            output.len(),
            number_of_states
        );
    }

    let ctx_device_id = config
        .ctx
        .device_id;
    if let Some(device_id) = states.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in input and context are different"
        );
    }
    if let Some(device_id) = output.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in output and context are different"
        );
    }
    check_device(ctx_device_id);
    let mut local_cfg = config.clone();
    local_cfg.are_states_on_device = states.is_on_device();
    local_cfg.are_outputs_on_device = output.is_on_device();

    <<F as FieldImpl>::Config as Poseidon2<F>>::poseidon_unchecked(
        states,
        output,
        number_of_states,
        width,
        constants,
        &local_cfg,
    )
}

#[macro_export]
macro_rules! impl_poseidon2 {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::poseidon2::{
                $field, $field_config, CudaError, DeviceContext, Poseidon2Config, Poseidon2Constants,
            };
            extern "C" {
                #[link_name = concat!($field_prefix, "_create_optimized_poseidon2_constants_cuda")]
                pub(crate) fn _create_optimized_constants(
                    width: u32,
                    alpha: u32,
                    external_rounds: u32,
                    internal_rounds: u32,
                    constants: *mut $field,
                    internal_matrix_diag: *mut $field,
                    ctx: &DeviceContext,
                    poseidon_constants: *mut Poseidon2Constants<$field>,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_init_optimized_poseidon2_constants_cuda")]
                pub(crate) fn _load_optimized_constants(
                    width: u32,
                    ctx: &DeviceContext,
                    constants: *mut Poseidon2Constants<$field>,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_hash_cuda")]
                pub(crate) fn hash_many(
                    states: *mut $field,
                    output: *mut $field,
                    number_of_states: u32,
                    width: u32,
                    constants: &Poseidon2Constants<$field>,
                    config: &Poseidon2Config,
                ) -> CudaError;
            }
        }

        impl Poseidon2<$field> for $field_config {
            fn create_optimized_constants<'a>(
                width: u32,
                alpha: u32,
                external_rounds: u32,
                internal_rounds: u32,
                round_constants: &mut [$field],
                internal_matrix_diag: &mut [$field],
                ctx: &DeviceContext,
            ) -> IcicleResult<Poseidon2Constants<'a, $field>> {
                unsafe {
                    let mut poseidon_constants = MaybeUninit::<Poseidon2Constants<'a, $field>>::uninit();
                    let err = $field_prefix_ident::_create_optimized_constants(
                        width,
                        alpha,
                        external_rounds,
                        internal_rounds,
                        round_constants as *mut _ as *mut $field,
                        internal_matrix_diag as *mut _ as *mut $field,
                        ctx,
                        poseidon_constants.as_mut_ptr(),
                    )
                    .wrap();
                    err.and(Ok(poseidon_constants.assume_init()))
                }
            }

            fn load_optimized_constants<'a>(
                width: u32,
                ctx: &DeviceContext,
            ) -> IcicleResult<Poseidon2Constants<'a, $field>> {
                unsafe {
                    let mut constants = MaybeUninit::<Poseidon2Constants<'a, $field>>::uninit();
                    let err = $field_prefix_ident::_load_optimized_constants(width, ctx, constants.as_mut_ptr()).wrap();

                    println!("Err: {:?}", err);
                    println!("Val: {:?}", constants.assume_init_read());
                    let t: Result<Poseidon2Constants<$field>, &str> = Ok(constants.assume_init_read());
                    println!("t.is_err = {}", t.is_err());
                    // println!("Ok(val) = {:?}", t);
                    // println!("Err + Val: {:?}", err.and(t) );
                    err.and(Ok(constants.assume_init()))
                }
            }

            fn poseidon_unchecked(
                states: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                number_of_states: u32,
                width: u32,
                constants: &Poseidon2Constants<$field>,
                config: &Poseidon2Config,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::hash_many(
                        states.as_mut_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        width,
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
macro_rules! impl_poseidon2_tests {
    (
      $field:ident
    ) => {
        #[test]
        fn test_poseidon2_hash_many() {
            check_poseidon_hash_many::<$field>()
        }
    };
}

#[macro_export]
macro_rules! impl_poseidon2_custom_config_test {
    (
      $field:ident,
      $field_bytes:literal,
      $field_prefix:literal,
      $partial_rounds:literal
    ) => {
        #[test]
        fn test_poseidon2_custom_config() {
            check_poseidon_custom_config::<$field>($field_bytes, $field_prefix, $partial_rounds)
        }
    };
}
