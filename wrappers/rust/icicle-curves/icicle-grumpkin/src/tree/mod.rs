use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeHash;
use icicle_core::impl_field_tree_builder;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{FieldTreeBuilder, TreeBuilderConfig};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use std::ffi::c_void;

use crate::curve::ScalarField;

impl_field_tree_builder!("grumpkin", grumpkin_tb, ScalarField, ScalarCfg, GrumpkinTreeBuilder);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::{ntt::FieldImpl, poseidon::Poseidon, tree::tests::check_build_field_merkle_tree};
    use icicle_cuda_runtime::device_context;

    use crate::curve::ScalarField;

    use super::GrumpkinTreeBuilder;

    #[test]
    fn poseidon_merkle_tree_test() {
        let ctx = device_context::DeviceContext::default();
        let sponge = Poseidon::load(2, &ctx).unwrap();

        check_build_field_merkle_tree::<_, _, GrumpkinTreeBuilder>(25, 2, &sponge, &sponge, ScalarField::zero());
    }
}
