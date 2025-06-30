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
        let field_element_size = std::mem::size_of::<ScalarField>() as u64;
        for elements_per_leaf in [1, 5] {
            // calc tree parameters
            let leaf_size = field_element_size * elements_per_leaf;
            let test_vec = ScalarCfg::generate_random(1 << n);
            let nof_leafs = (((1 << n) + elements_per_leaf - 1) / elements_per_leaf) as u64;
            let tree_height = if nof_leafs.is_power_of_two() {
                nof_leafs.ilog2() as usize
            } else {
                nof_leafs.ilog2() as usize + 1
            };

            let build_tree = |main_dev: bool| {
                if main_dev {
                    test_utilities::test_set_main_device();
                } else {
                    test_utilities::test_set_ref_device();
                }
                // define the tree
                let hasher = Blake2s::new(leaf_size).unwrap();
                let compress = Blake2s::new(hasher.output_size() * 2).unwrap();
                let layer_hashes: Vec<&Hasher> = std::iter::once(&hasher)
                    .chain(std::iter::repeat(&compress).take(tree_height))
                    .collect();

                println!("Leaf Size: {}", leaf_size);
                println!("Number of Leafs: {}", nof_leafs);
                println!("Hasher Output Size: {}", hasher.output_size());
                println!("Tree Height: {}", tree_height);

                let merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0).unwrap();
                let mut config = MerkleTreeConfig::default();
                config.padding_policy = PaddingPolicy::ZeroPadding;

                let build_start = Instant::now();

                merkle_tree
                    .build(HostSlice::from_slice(&test_vec), &config)
                    .unwrap();

                println!(
                    "Merkle tree build took: {:?} on device = {}",
                    build_start.elapsed(),
                    if main_dev { "Main" } else { "Ref" }
                );

                let root: &[u8] = merkle_tree
                    .get_root()
                    .unwrap();

                let mut root_vec = vec![0u8; hasher.output_size() as usize];
                root_vec.copy_from_slice(root);
                return root_vec;
            };

            // build the tree on main and ref devices and compare the roots
            let ref_root = build_tree(false);
            let main_root = build_tree(true);
            assert_eq!(ref_root, main_root);
        }
    }
}
