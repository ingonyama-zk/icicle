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
pub enum DiffusionStrategy {
    Default,
    Montgomery,
}

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

    pub mds_type: MdsType,

    pub diffusion: DiffusionStrategy,
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

    pub output_index: u32,

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
            output_index: 1,
            is_async: false,
        }
    }
}

pub trait Poseidon2<F: FieldImpl> {
    fn create_constants<'a>(
        width: u32,
        alpha: u32,
        internal_rounds: u32,
        external_rounds: u32,
        round_constants: &mut [F],
        internal_matrix_diag: &mut [F],
        mds_type: MdsType,
        diffusion: DiffusionStrategy,
        ctx: &DeviceContext,
    ) -> IcicleResult<Poseidon2Constants<'a, F>>;
    fn load_constants<'a>(
        width: u32,
        mds_type: MdsType,
        diffusion: DiffusionStrategy,
        ctx: &DeviceContext,
    ) -> IcicleResult<Poseidon2Constants<'a, F>>;
    fn poseidon_unchecked(
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        width: u32,
        constants: &Poseidon2Constants<F>,
        config: &Poseidon2Config,
    ) -> IcicleResult<()>;
    fn poseidon_unchecked_inplace(
        states: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        width: u32,
        constants: &Poseidon2Constants<F>,
        config: &Poseidon2Config,
    ) -> IcicleResult<()>;
    fn release_constants(constants: &Poseidon2Constants<F>, ctx: &DeviceContext) -> IcicleResult<()>;
}

/// Loads pre-calculated poseidon constants on the GPU.
pub fn load_poseidon2_constants<'a, F>(
    width: u32,
    mds_type: MdsType,
    diffusion: DiffusionStrategy,
    ctx: &DeviceContext,
) -> IcicleResult<Poseidon2Constants<'a, F>>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    <<F as FieldImpl>::Config as Poseidon2<F>>::load_constants(width, mds_type, diffusion, ctx)
}

/// Creates new instance of poseidon constants on the GPU.
pub fn create_poseidon2_constants<'a, F>(
    width: u32,
    alpha: u32,
    ctx: &DeviceContext,
    internal_rounds: u32,
    external_rounds: u32,
    round_constants: &mut [F],
    internal_matrix_diag: &mut [F],
    mds_type: MdsType,
    diffusion: DiffusionStrategy,
) -> IcicleResult<Poseidon2Constants<'a, F>>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    <<F as FieldImpl>::Config as Poseidon2<F>>::create_constants(
        width,
        alpha,
        internal_rounds,
        external_rounds,
        round_constants,
        internal_matrix_diag,
        mds_type,
        diffusion,
        ctx,
    )
}

fn poseidon_checks<F>(
    states: &(impl HostOrDeviceSlice<F> + ?Sized),
    output: &(impl HostOrDeviceSlice<F> + ?Sized),
    number_of_states: u32,
    width: u32,
    config: &Poseidon2Config,
) where
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
pub fn poseidon2_hash_many<F>(
    states: &(impl HostOrDeviceSlice<F> + ?Sized),
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
    poseidon_checks(states, output, number_of_states, width, config);
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

pub fn poseidon2_hash_many_inplace<F>(
    states: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    number_of_states: u32,
    width: u32,
    constants: &Poseidon2Constants<F>,
    config: &Poseidon2Config,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    poseidon_checks(states, states, number_of_states, width, config);
    let mut local_cfg = config.clone();
    local_cfg.are_states_on_device = states.is_on_device();
    local_cfg.are_outputs_on_device = states.is_on_device();

    <<F as FieldImpl>::Config as Poseidon2<F>>::poseidon_unchecked_inplace(
        states,
        number_of_states,
        width,
        constants,
        &local_cfg,
    )
}

pub fn release_poseidon2_constants<'a, F>(constants: &Poseidon2Constants<F>, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    <<F as FieldImpl>::Config as Poseidon2<F>>::release_constants(constants, ctx)
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
                $field, $field_config, CudaError, DeviceContext, DiffusionStrategy, MdsType, Poseidon2Config,
                Poseidon2Constants,
            };
            extern "C" {
                #[link_name = concat!($field_prefix, "_create_poseidon2_constants_cuda")]
                pub(crate) fn _create_constants(
                    width: u32,
                    alpha: u32,
                    internal_rounds: u32,
                    external_rounds: u32,
                    constants: *mut $field,
                    internal_matrix_diag: *mut $field,
                    mds_type: MdsType,
                    diffusion: DiffusionStrategy,
                    ctx: &DeviceContext,
                    poseidon_constants: *mut Poseidon2Constants<$field>,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_init_poseidon2_constants_cuda")]
                pub(crate) fn _load_constants(
                    width: u32,
                    mds_type: MdsType,
                    diffusion: DiffusionStrategy,
                    ctx: &DeviceContext,
                    constants: *mut Poseidon2Constants<$field>,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_release_poseidon2_constants_cuda")]
                pub(crate) fn _release_constants(
                    constants: &Poseidon2Constants<$field>,
                    ctx: &DeviceContext,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_hash_cuda")]
                pub(crate) fn hash_many(
                    states: *const $field,
                    output: *mut $field,
                    number_of_states: u32,
                    width: u32,
                    constants: &Poseidon2Constants<$field>,
                    config: &Poseidon2Config,
                ) -> CudaError;
            }
        }

        impl Poseidon2<$field> for $field_config {
            fn create_constants<'a>(
                width: u32,
                alpha: u32,
                internal_rounds: u32,
                external_rounds: u32,
                round_constants: &mut [$field],
                internal_matrix_diag: &mut [$field],
                mds_type: MdsType,
                diffusion: DiffusionStrategy,
                ctx: &DeviceContext,
            ) -> IcicleResult<Poseidon2Constants<'a, $field>> {
                unsafe {
                    let mut poseidon_constants = MaybeUninit::<Poseidon2Constants<'a, $field>>::uninit();
                    let err = $field_prefix_ident::_create_constants(
                        width,
                        alpha,
                        internal_rounds,
                        external_rounds,
                        round_constants as *mut _ as *mut $field,
                        internal_matrix_diag as *mut _ as *mut $field,
                        mds_type,
                        diffusion,
                        ctx,
                        poseidon_constants.as_mut_ptr(),
                    )
                    .wrap();
                    err.and(Ok(poseidon_constants.assume_init()))
                }
            }

            fn load_constants<'a>(
                width: u32,
                mds_type: MdsType,
                diffusion: DiffusionStrategy,
                ctx: &DeviceContext,
            ) -> IcicleResult<Poseidon2Constants<'a, $field>> {
                unsafe {
                    let mut constants = MaybeUninit::<Poseidon2Constants<'a, $field>>::uninit();
                    let err =
                        $field_prefix_ident::_load_constants(width, mds_type, diffusion, ctx, constants.as_mut_ptr())
                            .wrap();
                    err.and(Ok(constants.assume_init()))
                }
            }

            fn poseidon_unchecked(
                states: &(impl HostOrDeviceSlice<$field> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                number_of_states: u32,
                width: u32,
                constants: &Poseidon2Constants<$field>,
                config: &Poseidon2Config,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::hash_many(
                        states.as_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        width,
                        constants,
                        config,
                    )
                    .wrap()
                }
            }

            fn poseidon_unchecked_inplace(
                states: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                number_of_states: u32,
                width: u32,
                constants: &Poseidon2Constants<$field>,
                config: &Poseidon2Config,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::hash_many(
                        states.as_ptr(),
                        states.as_mut_ptr(),
                        number_of_states,
                        width,
                        constants,
                        config,
                    )
                    .wrap()
                }
            }

            fn release_constants<'a>(constants: &Poseidon2Constants<$field>, ctx: &DeviceContext) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_release_constants(constants, ctx).wrap() }
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

pub mod bench {
    use criterion::{black_box, Criterion};
    use icicle_cuda_runtime::{
        device_context::DeviceContext,
        memory::{HostOrDeviceSlice, HostSlice},
    };

    use crate::{
        ntt::FieldImpl,
        poseidon2::{load_poseidon2_constants, DiffusionStrategy, MdsType},
        traits::GenerateRandom,
        vec_ops::VecOps,
    };

    use super::{poseidon2_hash_many, Poseidon2, Poseidon2Config, Poseidon2Constants};

    #[allow(unused)]
    fn poseidon2_for_bench<'a, F: FieldImpl>(
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        poseidon2_result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: usize,
        width: usize,
        constants: &Poseidon2Constants<'a, F>,
        config: &Poseidon2Config,
        _seed: u32,
    ) where
        <F as FieldImpl>::Config: Poseidon2<F> + GenerateRandom<F>,
        <F as FieldImpl>::Config: VecOps<F>,
    {
        poseidon2_hash_many(
            states,
            poseidon2_result,
            number_of_states as u32,
            width as u32,
            constants,
            config,
        )
        .unwrap();
    }

    #[allow(unused)]
    pub fn benchmark_poseidon2<F: FieldImpl>(c: &mut Criterion)
    where
        <F as FieldImpl>::Config: Poseidon2<F> + GenerateRandom<F>,
        <F as FieldImpl>::Config: VecOps<F>,
    {
        use criterion::SamplingMode;
        use std::env;

        let group_id = format!("Poseidon2");
        let mut group = c.benchmark_group(&group_id);
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(10);

        const MAX_LOG2: u32 = 25; // max length = 2 ^ MAX_LOG2

        let max_log2 = env::var("MAX_LOG2")
            .unwrap_or_else(|_| MAX_LOG2.to_string())
            .parse::<u32>()
            .unwrap_or(MAX_LOG2);

        for test_size_log2 in 13u32..max_log2 + 1 {
            for t in [2, 3, 4, 8, 16, 20, 24] {
                let number_of_states = 1 << test_size_log2;
                let full_size = t * number_of_states;

                let scalars = F::Config::generate_random(full_size);
                let input = HostSlice::from_slice(&scalars);

                let mut permutation_result = vec![F::zero(); full_size];
                let permutation_result_slice = HostSlice::from_mut_slice(&mut permutation_result);

                let ctx = DeviceContext::default();
                let config = Poseidon2Config::default();
                for mds in [MdsType::Default, MdsType::Plonky] {
                    for diffusion in [DiffusionStrategy::Default, DiffusionStrategy::Montgomery] {
                        let constants =
                            load_poseidon2_constants(t as u32, mds.clone(), diffusion.clone(), &ctx).unwrap();
                        let bench_descr = format!(
                            "Mds::{:?}; Diffusion::{:?}; Number of states: {}; Width: {}",
                            mds, diffusion, number_of_states, t
                        );
                        group.bench_function(&bench_descr, |b| {
                            b.iter(|| {
                                poseidon2_for_bench::<F>(
                                    input,
                                    permutation_result_slice,
                                    number_of_states,
                                    t,
                                    &constants,
                                    &config,
                                    black_box(1),
                                )
                            })
                        });

                        // }
                    }
                }
            }
        }

        group.finish();
    }
}

#[macro_export]
macro_rules! impl_poseidon2_bench {
    (
      $field_prefix:literal,
      $field:ident
    ) => {
        use criterion::criterion_group;
        use criterion::criterion_main;
        use icicle_core::poseidon2::bench::benchmark_poseidon2;

        criterion_group!(benches, benchmark_poseidon2<$field>);
        criterion_main!(benches);
    };
}
