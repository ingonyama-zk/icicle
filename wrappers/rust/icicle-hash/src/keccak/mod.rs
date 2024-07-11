use icicle_core::hash::HashConfig;
use icicle_core::tree::TreeBuilderConfig;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use icicle_core::error::IcicleResult;
use icicle_core::traits::IcicleResultWrap;

pub mod tests;

extern "C" {
    pub(crate) fn keccak256_cuda(
        input: *const u8,
        input_block_size: i32,
        number_of_blocks: i32,
        output: *mut u8,
        config: &HashConfig,
    ) -> CudaError;

    pub(crate) fn keccak512_cuda(
        input: *const u8,
        input_block_size: i32,
        number_of_blocks: i32,
        output: *mut u8,
        config: &HashConfig,
    ) -> CudaError;

    pub(crate) fn build_keccak256_merkle_tree_cuda(
        leaves: *const u8,
        digests: *mut u64,
        height: u32,
        input_block_len: u32,
        config: &TreeBuilderConfig,
    ) -> CudaError;

    pub(crate) fn build_keccak512_merkle_tree_cuda(
        leaves: *const u8,
        digests: *mut u64,
        height: u32,
        input_block_len: u32,
        config: &TreeBuilderConfig,
    ) -> CudaError;
}

pub fn keccak256(
    input: &(impl HostOrDeviceSlice<u8> + ?Sized),
    input_block_size: i32,
    number_of_blocks: i32,
    output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: &mut HashConfig,
) -> IcicleResult<()> {
    config.are_inputs_on_device = input.is_on_device();
    config.are_outputs_on_device = output.is_on_device();
    unsafe {
        keccak256_cuda(
            input.as_ptr(),
            input_block_size,
            number_of_blocks,
            output.as_mut_ptr(),
            config,
        )
        .wrap()
    }
}

pub fn keccak512(
    input: &(impl HostOrDeviceSlice<u8> + ?Sized),
    input_block_size: i32,
    number_of_blocks: i32,
    output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: &mut HashConfig,
) -> IcicleResult<()> {
    config.are_inputs_on_device = input.is_on_device();
    config.are_outputs_on_device = output.is_on_device();
    unsafe {
        keccak512_cuda(
            input.as_ptr(),
            input_block_size,
            number_of_blocks,
            output.as_mut_ptr(),
            config,
        )
        .wrap()
    }
}

pub fn build_keccak256_merkle_tree(
    leaves: &(impl HostOrDeviceSlice<u8> + ?Sized),
    digests: &mut (impl HostOrDeviceSlice<u64> + ?Sized),
    height: usize,
    input_block_len: usize,
    config: &TreeBuilderConfig,
) -> IcicleResult<()> {
    unsafe {
        build_keccak256_merkle_tree_cuda(
            leaves.as_ptr(),
            digests.as_mut_ptr(),
            height as u32,
            input_block_len as u32,
            config,
        )
        .wrap()
    }
}

pub fn build_keccak512_merkle_tree(
    leaves: &(impl HostOrDeviceSlice<u8> + ?Sized),
    digests: &mut (impl HostOrDeviceSlice<u64> + ?Sized),
    height: usize,
    input_block_len: usize,
    config: &TreeBuilderConfig,
) -> IcicleResult<()> {
    unsafe {
        build_keccak512_merkle_tree_cuda(
            leaves.as_ptr(),
            digests.as_mut_ptr(),
            height as u32,
            input_block_len as u32,
            config,
        )
        .wrap()
    }
}
