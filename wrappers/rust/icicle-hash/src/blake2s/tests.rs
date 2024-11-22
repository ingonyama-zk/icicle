#[cfg(test)]
pub(crate) mod tests {
    use std::cmp::Reverse;

    use icicle_core::{
        hash::HashConfig,
        tree::{merkle_tree_digests_len, TreeBuilderConfig}, Matrix,
    };
    use icicle_cuda_runtime::memory::HostSlice;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use itertools::Itertools;

    use crate::blake2s::{blake2s, build_blake2s_merkle_tree, build_blake2s_mmcs};

    #[test]
    fn single_hash_test() {
        let config = HashConfig::default();

        let preimages = b"a";
        let mut digests = vec![0u8; 1 * 32];

        let preimages_slice = HostSlice::from_slice(preimages);
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        blake2s(
            preimages_slice,
            1 as u32,
            1 as u32,
            32 as u32,
            digests_slice,
            &config,
        )
        .unwrap();

        let hex_string: String = digests_slice.iter().map(|byte| format!("{:02x}", byte)).collect();

        assert_eq!(
            hex_string,
            "4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e90"
        );
    }

    #[test]
    fn blake2s_merkle_tree_test() {
        let mut config = TreeBuilderConfig::default();
        config.arity = 2;
        let height = 22;
        let input_block_len = 136;
        let leaves = vec![1u8; (1 << height) * input_block_len];
        let mut digests = vec![0u64; merkle_tree_digests_len((height + 1) as u32, 2, 1)];

        let leaves_slice = HostSlice::from_slice(&leaves);
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        build_blake2s_merkle_tree(leaves_slice, digests_slice, height, input_block_len, &config).unwrap();
        println!("Root: {:?}", digests_slice[0]);
    }

    #[test]
    fn blake2s_mmcs_test() {
        let mut config = TreeBuilderConfig::default();
        config.arity = 2;
        config.digest_elements = 32;
        let log_max = 4;
        let input_block_len = 4;

        let copy_matrices = 2;
        let mut matrices = vec![];

        let mut keep_leaves = vec![];
        for log in 1..log_max+1 {
            let mut leaves = vec![0u32; 1 << log];
            for i in 0..1<<log {
                leaves[i] = i as u32;
            }

            for _ in 0..copy_matrices {
                matrices.push(Matrix::from_slice(&leaves, input_block_len, 1 << log));
            }
            keep_leaves.push(leaves);
        }
        let digests_len = merkle_tree_digests_len(log_max as u32, 2, 32);
        let mut digests = vec![0u8; digests_len];
        let digests_slice = HostSlice::from_mut_slice(&mut digests);
        build_blake2s_mmcs(&matrices, digests_slice, &config).unwrap();
        assert_eq!(digests[0], 42);
        assert_eq!(digests[digests.len() - 1], 196);

        // for j in 0..digests_len / 32 {
        //     for i in 0..32 {
        //         print!("{:02x?}", digests[digests.len() - 32 * (digests_len / 32 - 1 - j) - 32 + i]);
        //     }
        //     println!();
        // }
    }

    #[test]
    fn blake2s_random_mmcs_test() {
        let mut config = TreeBuilderConfig::default();
        config.arity = 2;
        config.digest_elements = 32;

        const N_COLS: usize = 1;
        let log_size_range = 3..5;

        let mut rng = SmallRng::seed_from_u64(0);
        let log_sizes = (0..N_COLS)
            .map(|_| rng.gen_range(log_size_range.clone()))
            .collect_vec();
        let cols = log_sizes
            .iter()
            .map(|&log_size| {
                (0..(1 << log_size))
                    .map(|_| rng.gen_range(0u32..(1u32 << 30)))
                    .collect_vec()
            })
            .collect_vec();

        let mut matrices = vec![];
        for col in &cols {
            matrices.push(Matrix::from_slice(col, 4, col.len()));
        }

        let log_max = cols
            .iter()
            .sorted_by_key(|c| Reverse(c.len()))
            .next()
            .unwrap()
            .len()
            .ilog2();
        let digests_len = merkle_tree_digests_len(log_max as u32, 2, 32);
        let mut digests = vec![0u8; digests_len];
        let digests_slice = HostSlice::from_mut_slice(&mut digests);
        build_blake2s_mmcs(&matrices, digests_slice, &config).unwrap();

        for j in 0..digests_len / 32 {
            for i in 0..32 {
                print!("{:02x?}", digests[digests.len() - 32 * (digests_len / 32 - 1 - j) - 32 + i]);
            }
            println!();
        }
    }
}
