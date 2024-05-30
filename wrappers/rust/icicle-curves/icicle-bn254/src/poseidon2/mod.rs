use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeConfig;
use icicle_core::poseidon2::{DiffusionStrategy, MdsType, Poseidon2, Poseidon2Handle, Poseidon2Impl};
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{TreeBuilder, TreeBuilderConfig};
use icicle_core::{impl_poseidon2, impl_poseidon2_tree_builder};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::{DeviceSlice, HostOrDeviceSlice};

use core::mem::MaybeUninit;

impl_poseidon2!("bn254", bn254, ScalarField, ScalarCfg);
impl_poseidon2_tree_builder!("bn254", bn254_poseidon2_tb, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::Poseidon2TreeBuilder;
    use crate::curve::ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::ntt::FieldImpl;
    use icicle_core::poseidon2::{tests::*, DiffusionStrategy, MdsType, Poseidon2};
    use icicle_core::tree::tests::check_build_field_merkle_tree;
    use icicle_cuda_runtime::device_context;

    impl_poseidon2_tests!(ScalarField);

    #[test]
    fn poseidon2_merkle_tree_test() {
        let ctx = device_context::DeviceContext::default();
        let sponge = Poseidon2::load(2, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();

        check_build_field_merkle_tree::<_, _, Poseidon2TreeBuilder>(20, 2, &sponge, &sponge, ScalarField::zero());
    }

    #[test]
    fn test_poseidon2_kats() {
        let kats = [
            ScalarField::from_hex("0x0bb61d24daca55eebcb1929a82650f328134334da98ea4f847f760054f4a3033"),
            ScalarField::from_hex("0x303b6f7c86d043bfcbcc80214f26a30277a15d3f74ca654992defe7ff8d03570"),
            ScalarField::from_hex("0x1ed25194542b12eef8617361c3ba7c52e660b145994427cc86296242cf766ec8"),
        ];

        let poseidon = init_poseidon::<ScalarField>(3, MdsType::Default, DiffusionStrategy::Default);
        check_poseidon_kats(3, &kats, &poseidon);
    }
}
