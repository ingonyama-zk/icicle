#[doc(hidden)]
pub mod tests;

use std::{ffi::c_void, marker::PhantomData};

use icicle_cuda_runtime::{
    device::check_device,
    device_context::DeviceContext,
    memory::{HostOrDeviceSlice, HostSlice},
};

use crate::{error::IcicleResult, traits::FieldImpl};

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

    pub fn permute_many(
        &self,
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: usize,
        ctx: &DeviceContext,
    ) -> IcicleResult<()> {
        poseidon_checks(
            states,
            number_of_states * self.width,
            output,
            number_of_states * self.width,
            ctx,
        );
        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::permute_many(
            states,
            output,
            number_of_states as u32,
            self.handle,
            ctx,
        )
    }

    pub fn compress_many(
        &self,
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: usize,
        ctx: &DeviceContext,
    ) -> IcicleResult<()> {
        poseidon_checks(
            states,
            number_of_states as usize * self.width,
            output,
            number_of_states as usize,
            ctx,
        );
        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::compress_many(
            states,
            output,
            number_of_states as u32,
            0,
            self.handle,
            None::<&mut HostSlice<F>>,
            ctx,
        )
    }

    pub fn compress_many_advanced(
        &self,
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: usize,
        offset: u32,
        perm_output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        ctx: &DeviceContext,
    ) -> IcicleResult<()> {
        poseidon_checks(
            states,
            number_of_states as usize * self.width,
            output,
            number_of_states as usize,
            ctx,
        );
        <<F as FieldImpl>::Config as Poseidon2Impl<F>>::compress_many(
            states,
            output,
            number_of_states as u32,
            offset,
            self.handle,
            Some(perm_output),
            ctx,
        )
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
        round_constants: &mut [F],
        internal_matrix_diag: &mut [F],
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

    fn permute_many(
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        poseidon: Poseidon2Handle,
        ctx: &DeviceContext,
    ) -> IcicleResult<()>;

    fn compress_many(
        states: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        offset: u32,
        poseidon: Poseidon2Handle,
        perm_output: Option<&mut (impl HostOrDeviceSlice<F> + ?Sized)>,
        ctx: &DeviceContext,
    ) -> IcicleResult<()>;

    fn delete(poseidon: Poseidon2Handle, ctx: &DeviceContext) -> IcicleResult<()>;
}

fn poseidon_checks<F>(
    states: &(impl HostOrDeviceSlice<F> + ?Sized),
    states_size_expected: usize,
    output: &(impl HostOrDeviceSlice<F> + ?Sized),
    output_size_expected: usize,
    ctx: &DeviceContext,
) where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    if states.len() < states_size_expected {
        panic!(
            "input len is {}; but needs to be at least {}",
            states.len(),
            states_size_expected,
        );
    }
    if output.len() < output_size_expected {
        panic!(
            "output len is {}; but needs to be at least {}",
            output.len(),
            output_size_expected,
        );
    }

    let ctx_device_id = ctx.device_id;
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
                    constants: *mut $field,
                    internal_matrix_diag: *mut $field,
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

                #[link_name = concat!($field_prefix, "_poseidon2_permute_many_cuda")]
                pub(crate) fn permute_many(
                    poseidon: Poseidon2Handle,
                    states: *const $field,
                    output: *mut $field,
                    number_of_states: u32,
                    ctx: &DeviceContext,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon2_compress_many_cuda")]
                pub(crate) fn compress_many(
                    poseidon: Poseidon2Handle,
                    states: *const $field,
                    output: *mut $field,
                    number_of_states: u32,
                    offset: u32,
                    perm_output: *mut $field,
                    ctx: &DeviceContext,
                ) -> CudaError;
            }
        }

        impl Poseidon2Impl<$field> for $field_config {
            fn create(
                width: u32,
                alpha: u32,
                internal_rounds: u32,
                external_rounds: u32,
                round_constants: &mut [$field],
                internal_matrix_diag: &mut [$field],
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
                        round_constants as *mut _ as *mut $field,
                        internal_matrix_diag as *mut _ as *mut $field,
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

            fn permute_many(
                states: &(impl HostOrDeviceSlice<$field> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                number_of_states: u32,
                poseidon: Poseidon2Handle,
                ctx: &DeviceContext,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::permute_many(
                        poseidon,
                        states.as_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        ctx,
                    )
                    .wrap()
                }
            }

            fn compress_many(
                states: &(impl HostOrDeviceSlice<$field> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                number_of_states: u32,
                offset: u32,
                poseidon: Poseidon2Handle,
                perm_output: Option<&mut (impl HostOrDeviceSlice<$field> + ?Sized)>,
                ctx: &DeviceContext,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::compress_many(
                        poseidon,
                        states.as_ptr(),
                        output.as_mut_ptr(),
                        number_of_states,
                        offset,
                        match perm_output {
                            Some(perm_output) => perm_output.as_mut_ptr(),
                            None => std::ptr::null_mut(),
                        },
                        ctx,
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
        poseidon
            .permute_many(states, poseidon2_result, number_of_states, ctx)
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
