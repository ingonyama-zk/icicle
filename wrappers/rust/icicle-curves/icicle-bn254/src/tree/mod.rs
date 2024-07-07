pub mod mmcs;

use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeHash;
use icicle_core::impl_field_tree_builder;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{FieldTreeBuilder, TreeBuilderConfig};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use std::ffi::c_void;

use crate::curve::ScalarField;

impl_field_tree_builder!("bn254", bn254_tb, ScalarField, ScalarCfg, Bn254TreeBuilder);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::{
        ntt::FieldImpl,
        poseidon::Poseidon,
        poseidon2::{DiffusionStrategy, MdsType, Poseidon2},
        tree::tests::check_build_field_merkle_tree,
    };
    use icicle_cuda_runtime::device_context;

    use crate::curve::ScalarField;

    use super::Bn254TreeBuilder;

    #[test]
    fn poseidon_merkle_tree_test() {
        let ctx = device_context::DeviceContext::default();
        let sponge = Poseidon::load(2, &ctx).unwrap();

        check_build_field_merkle_tree::<_, _, Bn254TreeBuilder>(25, 2, &sponge, &sponge, ScalarField::zero());
    }

    #[test]
    fn poseidon2_merkle_tree_test() {
        let ctx = device_context::DeviceContext::default();
        let sponge = Poseidon2::load(2, 2, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();

        check_build_field_merkle_tree::<_, _, Bn254TreeBuilder>(28, 2, &sponge, &sponge, ScalarField::zero());
    }
}
