use icicle_core::hash::HashConfig;
use icicle_core::tree::TreeBuilderConfig;
use icicle_core::Matrix;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use icicle_core::error::IcicleResult;
use icicle_core::traits::IcicleResultWrap;

pub mod tests;

extern "C" {
    pub(crate) fn blake2s_cuda(
        input: *const u8,
        output: *mut u8,
        number_of_blocks: u32,
        input_block_size: u32,
        output_block_size: u32,
        config: &HashConfig,
    ) -> CudaError;

    pub(crate) fn build_blake2s_merkle_tree_cuda(
        leaves: *const u8,
        digests: *mut u64,
        height: u32,
        input_block_len: u32,
        tree_config: &TreeBuilderConfig,
    ) -> CudaError;

    pub(crate) fn blake2s_mmcs_commit_cuda(
        leaves: *const Matrix,
        number_of_inputs: u32,
        digests: *mut u8,
        tree_config: &TreeBuilderConfig
    ) -> CudaError;
}

pub fn blake2s(
    input: &(impl HostOrDeviceSlice<u8> + ?Sized),
    input_block_size: u32,
    number_of_blocks: u32,
    output_block_size: u32,
    output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: &HashConfig,
) -> IcicleResult<()> {
    let mut local_cfg = config.clone();
    local_cfg.are_inputs_on_device = input.is_on_device();
    local_cfg.are_outputs_on_device = output.is_on_device();
    unsafe {
        blake2s_cuda(
            input.as_ptr(),
            output.as_mut_ptr(),
            number_of_blocks,
            input_block_size,
            output_block_size,
            &local_cfg,
        )
        .wrap()
    }
}

pub fn build_blake2s_merkle_tree(
    leaves: &(impl HostOrDeviceSlice<u8> + ?Sized),
    digests: &mut (impl HostOrDeviceSlice<u64> + ?Sized),
    height: usize,
    input_block_len: usize,
    config: &TreeBuilderConfig,
) -> IcicleResult<()> {
    unsafe {
        build_blake2s_merkle_tree_cuda(
            leaves.as_ptr(),
            digests.as_mut_ptr(),
            height as u32,
            input_block_len as u32,
            config,
        )
        .wrap()
    }
}

pub fn build_blake2s_mmcs(
    leaves: &Vec<Matrix>,
    digests: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: &TreeBuilderConfig,
) -> IcicleResult<()> {
    unsafe {
        blake2s_mmcs_commit_cuda(
            leaves.as_ptr(),
            leaves.len() as u32,
            digests.as_mut_ptr(),
            config,
        )
        .wrap()
    }
}