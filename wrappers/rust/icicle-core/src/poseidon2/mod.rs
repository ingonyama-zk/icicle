#[doc(hidden)]
pub mod tests;

use std::{ffi::c_void, marker::PhantomData};

use icicle_cuda_runtime::{
    device_context::DeviceContext,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

use crate::{
    error::IcicleResult,
    hash::{sponge_check_input, sponge_check_outputs, sponge_check_states, SpongeConfig, SpongeHash},
    traits::FieldImpl,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum DiffusionStrategy {
    Default,
    Montgomery,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum MdsType {
    Default,
    Plonky,
}

pub type Poseidon2Handle = *const c_void;
pub struct Poseidon2<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    width: usize,
    handle: Poseidon2Handle,
    device_id: usize,
    phantom: PhantomData<F>,
}

impl<F> Poseidon2<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    pub fn load(
        width: usize,
        mds_type: MdsType,
        diffusion: DiffusionStrategy,
        ctx: &DeviceContext,
    ) -> IcicleResult<Self> {
        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::load(width as u32, mds_type, diffusion, ctx).and_then(
            |handle| {
                Ok(Self {
                    width,
                    handle,
                    device_id: ctx.device_id,
                    phantom: PhantomData,
                })
            },
        )
    }

    pub fn new(
        width: usize,
        alpha: u32,
        internal_rounds: u32,
        external_rounds: u32,
        round_constants: &mut [F],
        internal_matrix_diag: &mut [F],
        mds_type: MdsType,
        diffusion: DiffusionStrategy,
        ctx: &DeviceContext,
    ) -> IcicleResult<Self> {
        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::create(
            width as u32,
            alpha,
            internal_rounds,
            external_rounds,
            round_constants,
            internal_matrix_diag,
            mds_type,
            diffusion,
            ctx,
        )
        .and_then(|handle| {
            Ok(Self {
                width,
                handle,
                device_id: ctx.device_id,
                phantom: PhantomData,
            })
        })
    }
}

impl<F> SpongeHash<F, F> for Poseidon2<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    fn absorb_many(
        &self,
        inputs: &(impl HostOrDeviceSlice<F> + ?Sized),
        states: &mut DeviceSlice<F>,
        number_of_states: usize,
        input_block_len: usize,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()> {
        sponge_check_input(
            inputs,
            number_of_states,
            input_block_len,
            cfg.input_rate as usize,
            &cfg.ctx,
        );
        sponge_check_states(states, number_of_states, self.width, &cfg.ctx);

        let mut local_cfg = cfg.clone();
        local_cfg.are_inputs_on_device = inputs.is_on_device();

        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::absorb_many(
            inputs,
            states,
            number_of_states as u32,
            input_block_len as u32,
            self.handle,
            &local_cfg,
        )
    }

    fn squeeze_many(
        &self,
        states: &DeviceSlice<F>,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: usize,
        output_len: usize,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()> {
        sponge_check_outputs(
            output,
            number_of_states,
            output_len,
            self.width,
            cfg.recursive_squeeze,
            &cfg.ctx,
        );
        sponge_check_states(states, number_of_states, self.width, &cfg.ctx);

        let mut local_cfg = cfg.clone();
        local_cfg.are_outputs_on_device = output.is_on_device();

        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::squeeze_many(
            states,
            output,
            number_of_states as u32,
            output_len as u32,
            self.handle,
            &local_cfg,
        )
    }

    fn hash_many(
        &self,
        inputs: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: usize,
        input_block_len: usize,
        output_len: usize,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()> {
        sponge_check_input(
            inputs,
            number_of_states,
            input_block_len,
            cfg.input_rate as usize,
            &cfg.ctx,
        );
        sponge_check_outputs(output, number_of_states, output_len, self.width, false, &cfg.ctx);

        let mut local_cfg = cfg.clone();
        local_cfg.are_inputs_on_device = inputs.is_on_device();
        local_cfg.are_outputs_on_device = output.is_on_device();

        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::hash_many(
            inputs,
            output,
            number_of_states as u32,
            input_block_len as u32,
            output_len as u32,
            self.handle,
            &local_cfg,
        )
    }

    fn default_config<'a>(&self) -> SpongeConfig<'a> {
        let mut cfg = SpongeConfig::default();
        cfg.input_rate = self.width as u32;
        cfg.output_rate = self.width as u32;
        cfg
    }
}

impl<F> Drop for Poseidon2<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    fn drop(&mut self) {
        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::delete(
            self.handle,
            &DeviceContext::default_for_device(self.device_id),
        )
        .unwrap();
    }
}

pub trait Poseidon2Impl<F: FieldImpl> {
    fn create(
        width: u32,
        alpha: u32,
        internal_rounds: u32,
        external_rounds: u32,
        round_constants: &[F],
        internal_matrix_diag: &[F],
        mds_type: MdsType,
        diffusion: DiffusionStrategy,
        ctx: &DeviceContext,
    ) -> IcicleResult<Poseidon2Handle>;

    fn load(
        width: u32,
        mds_type: MdsType,
        diffusion: DiffusionStrategy,
        ctx: &DeviceContext,
    ) -> IcicleResult<Poseidon2Handle>;

    fn absorb_many(
        inputs: &(impl HostOrDeviceSlice<F> + ?Sized),
        states: &mut DeviceSlice<F>,
        number_of_states: u32,
        input_block_len: u32,
        poseidon: Poseidon2Handle,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn squeeze_many(
        states: &DeviceSlice<F>,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        output_len: u32,
        poseidon: Poseidon2Handle,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn hash_many(
        inputs: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        input_block_len: u32,
        output_len: u32,
        poseidon: Poseidon2Handle,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn delete(poseidon: Poseidon2Handle, ctx: &DeviceContext) -> IcicleResult<()>;
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
                $field, $field_config, CudaError, DeviceContext, DiffusionStrategy, MdsType, Poseidon2Handle,
                SpongeConfig,
            };
            use icicle_core::error::IcicleError;
            extern "C" {
                #[link_name = concat!($field_prefix, "_poseidon2_create_cuda")]
                pub(crate) fn create(
                    poseidon: *mut Poseidon2Handle,
                    width: u32,
                    alpha: u32,
                    internal_rounds: u32,
                    external_rounds: u32,
                    constants: *const $field,
                    internal_matrix_diag: *const $field,
                    mds_type: MdsType,
                    diffusion: DiffusionStrategy,
                    ctx: &DeviceContext,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_load_cuda")]
                pub(crate) fn load(
                    poseidon: *mut Poseidon2Handle,
                    width: u32,
                    mds_type: MdsType,
                    diffusion: DiffusionStrategy,
                    ctx: &DeviceContext,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_delete_cuda")]
                pub(crate) fn delete(poseidon: Poseidon2Handle, ctx: &DeviceContext) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_absorb_many_cuda")]
                pub(crate) fn absorb_many(
                    poseidon: Poseidon2Handle,
                    inputs: *const $field,
                    states: *mut $field,
                    number_of_states: u32,
                    input_block_len: u32,
                    cfg: &SpongeConfig,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_squeeze_many_cuda")]
                pub(crate) fn squeeze_many(
                    poseidon: Poseidon2Handle,
                    states: *const $field,
                    output: *mut $field,
                    number_of_states: u32,
                    output_len: u32,
                    cfg: &SpongeConfig,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_hash_many_cuda")]
                pub(crate) fn hash_many(
                    poseidon: Poseidon2Handle,
                    inputs: *const $field,
                    output: *mut $field,
                    number_of_states: u32,
                    input_block_len: u32,
                    output_len: u32,
                    cfg: &SpongeConfig,
                ) -> CudaError;
            }
        }

        impl Poseidon2Impl<$field> for $field_config {
            fn create(
                width: u32,
                alpha: u32,
                internal_rounds: u32,
                external_rounds: u32,
                round_constants: &[$field],
                internal_matrix_diag: &[$field],
                mds_type: MdsType,
                diffusion: DiffusionStrategy,
                ctx: &DeviceContext,
            ) -> IcicleResult<Poseidon2Handle> {
                unsafe {
                    let mut poseidon = MaybeUninit::<Poseidon2Handle>::uninit();
                    $field_prefix_ident::create(
                        poseidon.as_mut_ptr(),
                        width,
                        alpha,
                        internal_rounds,
                        external_rounds,
                        round_constants as *const _ as *const $field,
                        internal_matrix_diag as *const _ as *const $field,
                        mds_type,
                        diffusion,
                        ctx,
                    )
                    .wrap()
                    .and(Ok(poseidon.assume_init()))
                }
            }

            fn load(
                width: u32,
                mds_type: MdsType,
                diffusion: DiffusionStrategy,
                ctx: &DeviceContext,
            ) -> IcicleResult<Poseidon2Handle> {
                unsafe {
                    let mut poseidon = MaybeUninit::<Poseidon2Handle>::uninit();
                    $field_prefix_ident::load(poseidon.as_mut_ptr(), width, mds_type, diffusion, ctx)
                        .wrap()
                        .and(Ok(poseidon.assume_init()))
                }
            }

            fn absorb_many(
                inputs: &(impl HostOrDeviceSlice<$field> + ?Sized),
                states: &mut DeviceSlice<$field>,
                number_of_states: u32,
                input_block_len: u32,
                poseidon: Poseidon2Handle,
                cfg: &SpongeConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::absorb_many(
                        poseidon,
                        inputs.as_ptr(),
                        states.as_mut_ptr(),
                        number_of_states,
                        input_block_len,
                        cfg,
                    )
                    .wrap()
                }
            }

            fn squeeze_many(
                states: &DeviceSlice<$field>,
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                number_of_states: u32,
                output_len: u32,
                poseidon: Poseidon2Handle,
                cfg: &SpongeConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::squeeze_many(
                        poseidon,
                        states.as_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        output_len,
                        cfg,
                    )
                    .wrap()
                }
            }

            fn hash_many(
                inputs: &(impl HostOrDeviceSlice<$field> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                number_of_states: u32,
                input_block_len: u32,
                output_len: u32,
                poseidon: Poseidon2Handle,
                cfg: &SpongeConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::hash_many(
                        poseidon,
                        inputs.as_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        input_block_len,
                        output_len,
                        cfg,
                    )
                    .wrap()
                }
            }

            fn delete(poseidon: Poseidon2Handle, ctx: &DeviceContext) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::delete(poseidon, ctx).wrap() }
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
        hash::SpongeHash,
        ntt::FieldImpl,
        poseidon2::{DiffusionStrategy, MdsType, Poseidon2, Poseidon2Impl},
        traits::GenerateRandom,
        vec_ops::VecOps,
    };

    #[allow(unused)]
    fn poseidon2_for_bench<F: FieldImpl>(
        poseidon: &Poseidon2<F>,
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        poseidon2_result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: usize,
        ctx: &DeviceContext,
        _seed: u32,
    ) where
        <F as FieldImpl>::Config: Poseidon2Impl<F> + GenerateRandom<F>,
    {
        let cfg = poseidon.default_config();
        poseidon
            .hash_many(
                states,
                poseidon2_result,
                number_of_states,
                poseidon.width,
                poseidon.width,
                &cfg,
            )
            .unwrap();
    }

    #[allow(unused)]
    pub fn benchmark_poseidon2<F: FieldImpl>(c: &mut Criterion)
    where
        <F as FieldImpl>::Config: Poseidon2Impl<F> + GenerateRandom<F>,
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

        for test_size_log2 in 18u32..max_log2 + 1 {
            for t in [2, 3, 4, 8, 16, 20, 24] {
                let number_of_states = 1 << test_size_log2;
                let full_size = t * number_of_states;

                let scalars = F::Config::generate_random(full_size);
                let input = HostSlice::from_slice(&scalars);

                let mut permutation_result = vec![F::zero(); full_size];
                let permutation_result_slice = HostSlice::from_mut_slice(&mut permutation_result);

                let ctx = DeviceContext::default();
                for (mds, diffusion) in [
                    (MdsType::Default, DiffusionStrategy::Default),
                    (MdsType::Plonky, DiffusionStrategy::Montgomery),
                ] {
                    let poseidon = Poseidon2::<F>::load(t, mds, diffusion, &ctx).unwrap();
                    let bench_descr = format!(
                        "TestSize: 2**{}, Mds::{:?}, Diffusion::{:?}, Width: {}",
                        test_size_log2, mds, diffusion, t
                    );
                    group.bench_function(&bench_descr, |b| {
                        b.iter(|| {
                            poseidon2_for_bench::<F>(
                                &poseidon,
                                input,
                                permutation_result_slice,
                                number_of_states,
                                &ctx,
                                black_box(1),
                            )
                        })
                    });
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
