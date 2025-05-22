#[cfg(test)]
mod tests {

    use crate::field::{ScalarCfg, ScalarField};
    use icicle_core::{
        hash::Hasher,
        merkle::{MerkleTree, MerkleTreeConfig, PaddingPolicy},
        traits::GenerateRandom,
    };
    use icicle_hash::blake2s::Blake2s;
    use icicle_runtime::{device::Device, memory::HostSlice, runtime, test_utilities};
    use std::time::Instant;

    /// Initializes devices before running tests.
    pub fn initialize() {
        test_utilities::test_load_and_init_devices();
        test_utilities::test_set_main_device();
    }

    #[test]
    fn test_merkle_tree_segfault() {
        initialize();

        let n = 18;
        let test_vec = ScalarCfg::generate_random(1 << n);
        let leaf_size = std::mem::size_of::<ScalarField>() as u64;
        let nof_leafs = (1 << n) as u64;
        let hasher = Blake2s::new(leaf_size).unwrap();
        let compress = Blake2s::new(hasher.output_size() * 2).unwrap();

        let tree_height = nof_leafs.ilog2() as usize;
        let layer_hashes: Vec<&Hasher> = std::iter::once(&hasher)
            .chain(std::iter::repeat(&compress).take(tree_height))
            .collect();

        println!("Leaf Size: {}", leaf_size);
        println!("Number of Leafs: {}", 1 << n);
        println!("Hasher Output Size: {}", hasher.output_size());
        println!("Tree Height: {}", tree_height);

        let merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0).unwrap();

        let mut config = MerkleTreeConfig::default();
        config.padding_policy = PaddingPolicy::ZeroPadding;

        let build_start = Instant::now();

        // This call is expected to segfault for n=18 on CUDA
        merkle_tree
            .build(HostSlice::from_slice(&test_vec), &config)
            .unwrap();

        println!("Merkle tree build took: {:?}", build_start.elapsed());
    }
}
