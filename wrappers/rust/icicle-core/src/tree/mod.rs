use icicle_cuda_runtime::{
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::HostOrDeviceSlice,
};

use crate::error::IcicleResult;
use crate::hash::SpongeHash;

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
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
        }
    }
}

pub fn merkle_tree_digests_len(height: u32, arity: u32) -> usize {
    let mut digests_len = 0usize;
    let mut row_length = 1;
    for _ in 0..height {
        digests_len += row_length;
        row_length *= arity as usize;
    }
    digests_len
}

pub trait TreeBuilder<Compression, Sponge, Leaf, Digest>
where
    Compression: SpongeHash<Leaf, Digest>,
    Sponge: SpongeHash<Leaf, Digest>,
{
    fn build_merkle_tree(
        leaves: &(impl HostOrDeviceSlice<Leaf> + ?Sized),
        digests: &mut (impl HostOrDeviceSlice<Digest> + ?Sized),
        height: usize,
        input_block_len: usize,
        compression: &Compression,
        sponge: &Sponge,
        config: &TreeBuilderConfig,
    ) -> IcicleResult<()>;
}
