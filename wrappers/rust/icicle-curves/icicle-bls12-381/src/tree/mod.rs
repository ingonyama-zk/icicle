use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_tree_builder;
use icicle_core::poseidon::PoseidonConstants;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{TreeBuilder, TreeBuilderConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_tree_builder!("bls12_381", bls12_381, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_tree_builder_tests;
    use icicle_core::tree::tests::*;

    impl_tree_builder_tests!(ScalarField);
}
