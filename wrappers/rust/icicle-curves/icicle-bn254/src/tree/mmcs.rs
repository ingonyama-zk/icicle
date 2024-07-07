use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeHash;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{mmcs::FieldMmcs, TreeBuilderConfig};
use icicle_core::{impl_mmcs, Matrix};
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};
use std::ffi::c_void;

use crate::curve::ScalarField;

impl_mmcs!("bn254", bn254_mmcs, ScalarField, ScalarCfg, Bn254Mmcs);

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::c_void;

    use icicle_core::{
        poseidon2::{tests::init_poseidon, DiffusionStrategy, MdsType},
        traits::FieldImpl,
        tree::{merkle_tree_digests_len, TreeBuilderConfig},
        Matrix,
    };
    use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};

    use crate::{
        curve::ScalarField,
        tree::mmcs::{Bn254Mmcs, FieldMmcs},
    };

    #[test]
    fn test_poseidon2_mmcs() {
        const WIDTH: usize = 3;
        const ARITY: usize = 2;
        const HEIGHT: usize = 20;
        const ROWS: usize = 1 << HEIGHT;
        const COLS: usize = 300;
        const DIGEST_ELEMENTS: usize = 1;

        let poseidon = init_poseidon::<ScalarField>(WIDTH, MdsType::Default, DiffusionStrategy::Default);

        let mut icicle_input = vec![ScalarField::zero(); ROWS * COLS];
        for i in 0..ROWS * COLS {
            icicle_input[i] = ScalarField::one();
        }

        let mut config = TreeBuilderConfig::default();
        config.arity = ARITY as u32;
        config.keep_rows = HEIGHT as u32 + 1;
        config.digest_elements = DIGEST_ELEMENTS as u32;
        let digests_len = merkle_tree_digests_len(HEIGHT as u32, ARITY as u32, DIGEST_ELEMENTS as u32);
        let mut digests = vec![ScalarField::zero(); digests_len];

        let leaves_slice = vec![Matrix {
            values: icicle_input.as_ptr() as *const c_void,
            width: COLS,
            height: ROWS,
        }];
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        let start = std::time::Instant::now();
        Bn254Mmcs::mmcs_commit(leaves_slice, digests_slice, &poseidon, &poseidon, &config).unwrap();
        println!("time: {:?}", start.elapsed());

        for i in 0..DIGEST_ELEMENTS {
            println!("{}", digests_slice[digests_slice.len() - DIGEST_ELEMENTS + i]);
        }
    }
}
