use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeHash;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{mmcs::FieldMmcs, TreeBuilderConfig};
use icicle_core::{impl_mmcs, Matrix};
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};
use std::ffi::c_void;

use crate::field::ScalarField;

impl_mmcs!("babybear", babybear_mmcs, ScalarField, ScalarCfg, BabyBearMmcs);

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::c_void;

    use icicle_core::{
        ntt::FieldImpl,
        tree::{merkle_tree_digests_len, TreeBuilderConfig},
        Matrix,
    };
    use icicle_cuda_runtime::memory::HostSlice;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_commit::Mmcs;
    use p3_field::{AbstractField, Field};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2 as PlonkyPoseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use crate::{
        field::ScalarField,
        poseidon2::tests::get_plonky3_poseidon2_t16,
        tree::mmcs::{BabyBearMmcs, FieldMmcs},
    };

    type PlonkyPoseidon2T16 = PlonkyPoseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;

    #[test]
    fn test_poseidon2_mmcs_plonky3() {
        const WIDTH: usize = 16;
        const RATE: usize = 8;
        const ARITY: usize = 2;
        const HEIGHT: usize = 15;
        const ROWS: usize = 1 << HEIGHT;
        const COLS: usize = 32;
        const DIGEST_ELEMENTS: usize = 8;

        let (poseidon, plonky_poseidon2) = get_plonky3_poseidon2_t16(RATE);

        type H = PaddingFreeSponge<PlonkyPoseidon2T16, WIDTH, RATE, RATE>;
        let h = H::new(plonky_poseidon2.clone());

        type C = TruncatedPermutation<PlonkyPoseidon2T16, ARITY, RATE, WIDTH>;
        let c = C::new(plonky_poseidon2.clone());

        type F = BabyBear;

        let mut input = vec![F::zero(); ROWS * COLS];
        let mut icicle_input = vec![ScalarField::zero(); ROWS * COLS];
        for i in 0..ROWS * COLS {
            input[i] = F::from_canonical_u32(i as u32);
            icicle_input[i] = ScalarField::from_u32(i as u32);
        }

        let mut input2 = vec![F::zero(); (ROWS / 2) * COLS];
        let mut icicle_input2 = vec![ScalarField::zero(); (ROWS / 2) * COLS];
        for i in 0..(ROWS / 2) * COLS {
            input2[i] = F::from_canonical_u32(i as u32);
            icicle_input2[i] = ScalarField::from_u32(i as u32);
        }

        let matrix = RowMajorMatrix::new(input.clone(), COLS);
        let matrix2 = RowMajorMatrix::new(input2.clone(), COLS);
        // let leaves = vec![matrix, matrix2];
        let leaves = vec![matrix];

        let mmcs =
            FieldMerkleTreeMmcs::<<F as Field>::Packing, <F as Field>::Packing, H, C, DIGEST_ELEMENTS>::new(h, c);

        let (commit, _data) = mmcs.commit(leaves);

        let mut config = TreeBuilderConfig::default();
        config.arity = ARITY as u32;
        config.keep_rows = HEIGHT as u32 + 1;
        config.digest_elements = DIGEST_ELEMENTS as u32;
        let digests_len = merkle_tree_digests_len(HEIGHT as u32, ARITY as u32, DIGEST_ELEMENTS as u32);
        println!("Digests len: {}", digests_len);
        let mut digests = vec![ScalarField::zero(); digests_len];
        // let mut digests = vec![ScalarField::zero(); COLS];

        let leaves_slice = vec![
            Matrix {
                values: icicle_input.as_ptr() as *const c_void,
                width: COLS,
                height: ROWS,
            },
            // Matrix {
            //     values: icicle_input2.as_ptr() as *const c_void,
            //     width: COLS,
            //     height: ROWS / 2,
            // },
        ];
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        BabyBearMmcs::mmcs_commit(leaves_slice, digests_slice, &poseidon, &poseidon, &config).unwrap();

        let mut converted = vec![BabyBear::zero(); digests_len];
        for i in 0..digests_len {
            let mut scalar_bytes = [0u8; 4];
            scalar_bytes.copy_from_slice(&digests_slice[i].to_bytes_le());
            converted[i] = BabyBear::from_canonical_u32(u32::from_le_bytes(scalar_bytes));
        }

        // println!("Plonky: {:?}", _data);
        // println!("Icicle: {:?}", converted);
        // assert_eq!(commit, converted);

        let commit_vec: Vec<BabyBear> = commit
            .into_iter()
            .collect();
        for i in 0..DIGEST_ELEMENTS {
            assert_eq!(converted[converted.len() - DIGEST_ELEMENTS + i], commit_vec[i]);
        }
    }
}
