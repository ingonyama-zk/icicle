use icicle_cuda_runtime::{
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::HostOrDeviceSlice,
};

use crate::hash::SpongeHash;
use crate::{error::IcicleResult, ntt::FieldImpl};

pub mod mmcs;
#[doc(hidden)]
pub mod tests;

/// Struct that encodes Tree Builder parameters to be passed into the [build_merkle_tree](build_merkle_tree) function.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TreeBuilderConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,

    /// Airty of the tree
    pub arity: u32,

    /// How many rows of the Merkle tree rows should be written to output. '0' means all of them
    pub keep_rows: u32,

    /// The size of the output for the bottom layer hash and compression.
    /// Will also be equal to the size of the root of the tree. Default value 1
    pub digest_elements: u32,

    are_inputs_on_device: bool,

    are_outputs_on_device: bool,

    /// Whether to run build_merkle_tree asynchronously. If set to `true`, TreeBuilder will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, build_merkle_tree will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for TreeBuilderConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> TreeBuilderConfig<'a> {
    fn default_for_device(device_id: usize) -> Self {
        Self {
            ctx: DeviceContext::default_for_device(device_id),
            arity: 2,
            keep_rows: 0,
            digest_elements: 1,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
        }
    }
}

pub fn merkle_tree_digests_len(height: u32, arity: u32, digest_elements: u32) -> usize {
    let mut digests_len = 0usize;
    let mut row_length = digest_elements as usize;
    for _ in 0..height + 1 {
        digests_len += row_length;
        row_length *= arity as usize;
    }
    digests_len
}

pub trait FieldTreeBuilder<F, Compression, Sponge>
where
    F: FieldImpl,
    Compression: SpongeHash<F, F>,
    Sponge: SpongeHash<F, F>,
{
    fn build_merkle_tree(
        leaves: &(impl HostOrDeviceSlice<F> + ?Sized),
        digests: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        height: usize,
        input_block_len: usize,
        compression: &Compression,
        sponge: &Sponge,
        config: &TreeBuilderConfig,
    ) -> IcicleResult<()>;
}

#[macro_export]
macro_rules! impl_field_tree_builder {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident,
      $tree_builder:ident
    ) => {
        mod $field_prefix_ident {
            use super::*;
            use icicle_cuda_runtime::error::CudaError;

            extern "C" {
                #[link_name = concat!($field_prefix, "_build_merkle_tree")]
                pub(crate) fn build_merkle_tree(
                    leaves: *const $field,
                    digests: *mut $field,
                    height: u32,
                    input_block_len: u32,
                    compression: *const c_void,
                    sponge: *const c_void,
                    config: &TreeBuilderConfig,
                ) -> CudaError;
            }
        }

        struct $tree_builder;

        impl<Compression, Sponge> FieldTreeBuilder<$field, Compression, Sponge> for $tree_builder
        where
            Compression: SpongeHash<$field, $field>,
            Sponge: SpongeHash<$field, $field>,
        {
            fn build_merkle_tree(
                leaves: &(impl HostOrDeviceSlice<$field> + ?Sized),
                digests: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                height: usize,
                input_block_len: usize,
                compression: &Compression,
                sponge: &Sponge,
                config: &TreeBuilderConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::build_merkle_tree(
                        leaves.as_ptr(),
                        digests.as_mut_ptr(),
                        height as u32,
                        input_block_len as u32,
                        compression.get_handle(),
                        sponge.get_handle(),
                        config,
                    )
                    .wrap()
                }
            }
        }
    };
}
