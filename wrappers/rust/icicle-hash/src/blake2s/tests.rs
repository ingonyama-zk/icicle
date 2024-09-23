#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::{
        hash::HashConfig,
        tree::{merkle_tree_digests_len, TreeBuilderConfig},
    };
    use icicle_cuda_runtime::memory::HostSlice;

    use crate::blake2s::{build_blake2s_merkle_tree, blake2s};

    #[test]
    fn blake2s_hash_test() {
        let config = HashConfig::default();
        let input_block_len = 136;
        let number_of_hashes = 1024;
        let output_block_len = 32;

        let preimages = vec![1u8; number_of_hashes * input_block_len];
        let mut digests = vec![0u8; number_of_hashes * 32];

        let preimages_slice = HostSlice::from_slice(&preimages);
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        blake2s(
            preimages_slice,
            input_block_len as u32,
            number_of_hashes as u32,
            output_block_len as u32,
            digests_slice,
            &config,
        )
        .unwrap();
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
}
