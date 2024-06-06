#[doc(hidden)]
pub mod tests;

pub mod tree;

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

pub type PoseidonHandle = *const c_void;
pub struct Poseidon<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonImpl<F>,
{
    width: usize,
    pub handle: PoseidonHandle,
    phantom: PhantomData<F>,
}

impl<F> Poseidon<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonImpl<F>,
{
    pub fn load(arity: usize, ctx: &DeviceContext) -> IcicleResult<Self> {
        <<F as FieldImpl>::Config as PoseidonImpl<F>>::load(arity as u32, ctx).and_then(|handle| {
            Ok(Self {
                width: arity + 1,
                handle,
                phantom: PhantomData,
            })
        })
    }

    pub fn new(
        arity: usize,
        alpha: u32,
        full_rounds_half: u32,
        partial_rounds: u32,
        round_constants: &[F],
        mds_matrix: &[F],
        non_sparse_matrix: &[F],
        sparse_matrices: &[F],
        domain_tag: F,
        ctx: &DeviceContext,
    ) -> IcicleResult<Self> {
        <<F as FieldImpl>::Config as PoseidonImpl<F>>::create(
            arity as u32,
            alpha,
            full_rounds_half,
            partial_rounds,
            round_constants,
            mds_matrix,
            non_sparse_matrix,
            sparse_matrices,
            domain_tag,
            ctx,
        )
        .and_then(|handle| {
            Ok(Self {
                width: arity + 1,
                handle,
                phantom: PhantomData,
            })
        })
    }
}

impl<F> SpongeHash<F, F> for Poseidon<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonImpl<F>,
{
    fn absorb_many(
        &self,
        inputs: &(impl HostOrDeviceSlice<F> + ?Sized),
        states: &mut DeviceSlice<F>,
        number_of_states: usize,
        input_block_len: usize,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()> {
        sponge_check_input(inputs, number_of_states, input_block_len, self.width - 1, &cfg.ctx);
        sponge_check_states(states, number_of_states, self.width, &cfg.ctx);

        let mut local_cfg = cfg.clone();
        local_cfg.are_inputs_on_device = inputs.is_on_device();

        <<F as FieldImpl>::Config as PoseidonImpl<F>>::absorb_many(
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

        <<F as FieldImpl>::Config as PoseidonImpl<F>>::squeeze_many(
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
        sponge_check_input(inputs, number_of_states, input_block_len, self.width - 1, &cfg.ctx);
        sponge_check_outputs(output, number_of_states, output_len, self.width, false, &cfg.ctx);

        let mut local_cfg = cfg.clone();
        local_cfg.are_inputs_on_device = inputs.is_on_device();
        local_cfg.are_outputs_on_device = output.is_on_device();

        <<F as FieldImpl>::Config as PoseidonImpl<F>>::hash_many(
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
        cfg.input_rate = self.width as u32 - 1;
        cfg.output_rate = self.width as u32;
        cfg
    }
}

impl<F> Drop for Poseidon<F>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonImpl<F>,
{
    fn drop(&mut self) {
        <<F as FieldImpl>::Config as PoseidonImpl<F>>::delete(
            self.handle,
        )
        .unwrap();
    }
}

pub trait PoseidonImpl<F: FieldImpl> {
    fn create(
        arity: u32,
        alpha: u32,
        full_rounds_half: u32,
        partial_rounds: u32,
        round_constants: &[F],
        mds_matrix: &[F],
        non_sparse_matrix: &[F],
        sparse_matrices: &[F],
        domain_tag: F,
        ctx: &DeviceContext,
    ) -> IcicleResult<PoseidonHandle>;

    fn load(arity: u32, ctx: &DeviceContext) -> IcicleResult<PoseidonHandle>;

    fn absorb_many(
        inputs: &(impl HostOrDeviceSlice<F> + ?Sized),
        states: &mut DeviceSlice<F>,
        number_of_states: u32,
        input_block_len: u32,
        poseidon: PoseidonHandle,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn squeeze_many(
        states: &DeviceSlice<F>,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        output_len: u32,
        poseidon: PoseidonHandle,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn hash_many(
        inputs: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        number_of_states: u32,
        input_block_len: u32,
        output_len: u32,
        poseidon: PoseidonHandle,
        cfg: &SpongeConfig,
    ) -> IcicleResult<()>;

    fn delete(poseidon: PoseidonHandle) -> IcicleResult<()>;
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
            use crate::poseidon::{$field, $field_config, CudaError, DeviceContext, PoseidonHandle, SpongeConfig};
            extern "C" {
                #[link_name = concat!($field_prefix, "_poseidon_create_cuda")]
                pub(crate) fn create(
                    poseidon: *mut PoseidonHandle,
                    arity: u32,
                    alpha: u32,
                    full_rounds_half: u32,
                    partial_rounds: u32,
                    round_constants: *const $field,
                    mds_matrix: *const $field,
                    non_sparse_matrix: *const $field,
                    sparse_matrices: *const $field,
                    domain_tag: $field,
                    ctx: &DeviceContext,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon_load_cuda")]
                pub(crate) fn load(poseidon: *mut PoseidonHandle, arity: u32, ctx: &DeviceContext) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon_delete_cuda")]
                pub(crate) fn delete(poseidon: PoseidonHandle) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon_absorb_many_cuda")]
                pub(crate) fn absorb_many(
                    poseidon: PoseidonHandle,
                    inputs: *const $field,
                    states: *mut $field,
                    number_of_states: u32,
                    input_block_len: u32,
                    cfg: &SpongeConfig,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon_squeeze_many_cuda")]
                pub(crate) fn squeeze_many(
                    poseidon: PoseidonHandle,
                    states: *const $field,
                    output: *mut $field,
                    number_of_states: u32,
                    output_len: u32,
                    cfg: &SpongeConfig,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_poseidon_hash_many_cuda")]
                pub(crate) fn hash_many(
                    poseidon: PoseidonHandle,
                    inputs: *const $field,
                    output: *mut $field,
                    number_of_states: u32,
                    input_block_len: u32,
                    output_len: u32,
                    cfg: &SpongeConfig,
                ) -> CudaError;
            }
        }

        impl PoseidonImpl<$field> for $field_config {
            fn create(
                arity: u32,
                alpha: u32,
                full_rounds_half: u32,
                partial_rounds: u32,
                round_constants: &[$field],
                mds_matrix: &[$field],
                non_sparse_matrix: &[$field],
                sparse_matrices: &[$field],
                domain_tag: $field,
                ctx: &DeviceContext,
            ) -> IcicleResult<PoseidonHandle> {
                unsafe {
                    let mut poseidon = MaybeUninit::<PoseidonHandle>::uninit();
                    $field_prefix_ident::create(
                        poseidon.as_mut_ptr(),
                        arity,
                        alpha,
                        full_rounds_half,
                        partial_rounds,
                        round_constants as *const _ as *const $field,
                        mds_matrix as *const _ as *const $field,
                        non_sparse_matrix as *const _ as *const $field,
                        sparse_matrices as *const _ as *const $field,
                        domain_tag,
                        ctx,
                    )
                    .wrap()
                    .and(Ok(poseidon.assume_init()))
                }
            }

            fn load(arity: u32, ctx: &DeviceContext) -> IcicleResult<PoseidonHandle> {
                unsafe {
                    let mut poseidon = MaybeUninit::<PoseidonHandle>::uninit();
                    $field_prefix_ident::load(poseidon.as_mut_ptr(), arity, ctx)
                        .wrap()
                        .and(Ok(poseidon.assume_init()))
                }
            }

            fn absorb_many(
                inputs: &(impl HostOrDeviceSlice<$field> + ?Sized),
                states: &mut DeviceSlice<$field>,
                number_of_states: u32,
                input_block_len: u32,
                poseidon: PoseidonHandle,
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
                poseidon: PoseidonHandle,
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
                poseidon: PoseidonHandle,
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

            fn delete(poseidon: PoseidonHandle) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::delete(poseidon).wrap() }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_poseidon_tests {
    (
      $field:ident
    ) => {
        #[test]
        fn test_poseidon_hash_many() {
            check_poseidon_hash_many::<$field>()
        }
    };
}
