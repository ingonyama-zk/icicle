use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeHash;
use icicle_core::impl_field_tree_builder;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{FieldTreeBuilder, TreeBuilderConfig};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use std::ffi::c_void;

use crate::field::ScalarField;

pub mod mmcs;

impl_field_tree_builder!("babybear", babybear_tb, ScalarField, ScalarCfg, BabyBearTreeBuilder);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::{
        ntt::FieldImpl,
        poseidon2::{DiffusionStrategy, MdsType, Poseidon2},
        tree::{tests::check_build_field_merkle_tree, FieldTreeBuilder, TreeBuilderConfig},
    };
    use icicle_cuda_runtime::device_context;
    use icicle_cuda_runtime::memory::HostSlice;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_commit::Mmcs;
    use p3_field::{AbstractField, Field};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2 as PlonkyPoseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use crate::{field::ScalarField, poseidon2::tests::get_plonky3_poseidon2_t16, tree::BabyBearTreeBuilder};

    #[test]
    fn poseidon2_merkle_tree_test() {
        let ctx = device_context::DeviceContext::default();
        let sponge = Poseidon2::load(2, 2, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();

        check_build_field_merkle_tree::<_, _, BabyBearTreeBuilder>(25, 2, &sponge, &sponge, ScalarField::zero());
    }

    type PlonkyPoseidon2T16 = PlonkyPoseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;

    #[test]
    fn test_poseidon2_tree_plonky3() {
        const WIDTH: usize = 16;
        const ARITY: usize = 2;
        const HEIGHT: usize = 15;
        const ROWS: usize = 1 << HEIGHT;
        const COLS: usize = 8;

        let (poseidon, plonky_poseidon2) = get_plonky3_poseidon2_t16(8);

        type H = PaddingFreeSponge<PlonkyPoseidon2T16, WIDTH, COLS, COLS>;
        let h = H::new(plonky_poseidon2.clone());

        type C = TruncatedPermutation<PlonkyPoseidon2T16, ARITY, COLS, WIDTH>;
        let c = C::new(plonky_poseidon2.clone());

        type F = BabyBear;

        let mut input = vec![F::zero(); ROWS * COLS];
        let mut icicle_input = vec![ScalarField::zero(); ROWS * COLS];
        for i in 0..ROWS * COLS {
            input[i] = F::from_canonical_u32(i as u32);
            icicle_input[i] = ScalarField::from_u32(i as u32);
        }

        let matrix = RowMajorMatrix::new(input, COLS);
        let leaves = vec![matrix];

        let mmcs = FieldMerkleTreeMmcs::<<F as Field>::Packing, <F as Field>::Packing, H, C, 8>::new(h, c);

        let (commit, _data) = mmcs.commit(leaves);

        let mut config = TreeBuilderConfig::default();
        config.arity = ARITY as u32;
        config.keep_rows = 1;
        config.digest_elements = COLS as u32;
        let input_block_len = COLS;
        // let digests_len = merkle_tree_digests_len(2 as u32, ARITY as u32, COLS as u32);
        // let mut digests = vec![ScalarField::zero(); digests_len];
        let mut digests = vec![ScalarField::zero(); COLS];

        let leaves_slice = HostSlice::from_slice(&icicle_input);
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        BabyBearTreeBuilder::build_merkle_tree(
            leaves_slice,
            digests_slice,
            HEIGHT,
            input_block_len,
            &poseidon,
            &poseidon,
            &config,
        )
        .unwrap();

        let mut converted: [BabyBear; COLS] = [BabyBear::zero(); COLS];
        for i in 0..COLS {
            let mut scalar_bytes = [0u8; 4];
            scalar_bytes.copy_from_slice(&digests_slice[i].to_bytes_le());
            converted[i] = BabyBear::from_canonical_u32(u32::from_le_bytes(scalar_bytes));
        }
        assert_eq!(commit, converted);
    }
}
