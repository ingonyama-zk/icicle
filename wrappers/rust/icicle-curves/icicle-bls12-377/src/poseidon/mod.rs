#[cfg(feature = "bw6-761")]
use crate::curve::{BaseCfg, BaseField};
use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeConfig;
use icicle_core::poseidon::{Poseidon, PoseidonHandle, PoseidonImpl};
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{TreeBuilder, TreeBuilderConfig};
use icicle_core::{impl_poseidon, impl_poseidon_tree_builder};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::{DeviceSlice, HostOrDeviceSlice};

use core::mem::MaybeUninit;

impl_poseidon!("bls12_377", bls12_377, ScalarField, ScalarCfg);
impl_poseidon_tree_builder!("bls12_377", bls12_377_tb, ScalarField, ScalarCfg);

#[cfg(feature = "bw6-761")]
impl_poseidon!("bw6_761", bw6_761, BaseField, BaseCfg);
impl_poseidon_tree_builder!("bw6_761", bw6_761, BaseField, BaseCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::PoseidonTreeBuilder;
    use crate::curve::ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::ntt::FieldImpl;
    use icicle_core::poseidon::{tests::*, Poseidon};
    use icicle_core::tree::tests::*;
    use icicle_cuda_runtime::device_context;

    impl_poseidon_tests!(ScalarField);

    #[test]
    fn poseidon_merkle_tree_test() {
        let ctx = device_context::DeviceContext::default();
        let sponge = Poseidon::load(2, &ctx).unwrap();

        check_build_field_merkle_tree::<_, _, PoseidonTreeBuilder>(20, 2, &sponge, &sponge, ScalarField::zero());
    }
}
