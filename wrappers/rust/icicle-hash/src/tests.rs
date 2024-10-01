#[cfg(test)]
mod tests {

    use crate::{
        blake2s::Blake2s,
        keccak::{Keccak256, Keccak512},
        sha3::{Sha3_256, Sha3_512},
    };
    use icicle_core::{
        hash::{HashConfig, Hasher},
        merkle::{MerkleProof, MerkleTree, MerkleTreeConfig},
        test_utilities,
    };
    use icicle_runtime::memory::HostSlice;
    use rand::Rng;
    use std::sync::Once;

    static INIT: Once = Once::new();

    pub fn initialize() {
        INIT.call_once(move || {
            // TODO load CUDA backend
            // test_utilities::test_load_and_init_devices();
        });
    }

    #[test]
    fn keccak_hashing() {
        initialize();
        test_utilities::test_set_ref_device();
        let single_hash_input_size = 30;
        let batch = 3;
        let keccak_hasher = Keccak512::new(0 /*default chunk size */).unwrap();
        let input = vec![0 as u8; single_hash_input_size * batch];
        let mut output = vec![0 as u8; 64 * batch]; // 64B (=512b) is the output size of Keccak512,
        keccak_hasher
            .hash(
                HostSlice::from_slice(&input),
                &HashConfig::default(),
                HostSlice::from_mut_slice(&mut output),
            )
            .unwrap();
        println!("output= {:?}", output);
        // TODO compare to main device (CUDA by default) or verify with goldens
    }

    #[test]
    fn blake2s_hashing() {
        initialize();
        test_utilities::test_set_ref_device();
        let single_hash_input_size = 8;
        let batch = 3;
        let blake2s_hasher = Blake2s::new(0 /*default chunk size */).unwrap();
        let input = vec![0 as u8; single_hash_input_size * batch];
        let mut output = vec![0 as u8; 8 * batch]; // 8B (=64b) is the output size of Blake2s,
        blake2s_hasher
            .hash(
                HostSlice::from_slice(&input),
                &HashConfig::default(),
                HostSlice::from_mut_slice(&mut output),
            )
            .unwrap();
        println!("output= {:?}", output);
        // TODO compare to main device (CUDA by default) or verify with goldens
    }

    #[test]
    fn sha3_hashing() {
        initialize();
        test_utilities::test_set_ref_device();

        let sha3_hasher = Sha3_512::new(0 /*default chunk size */).unwrap();
        let input = vec![0 as u8; 90];
        let mut output = vec![0 as u8; 64]; // 256b * batch
        sha3_hasher
            .hash(
                HostSlice::from_slice(&input),
                &HashConfig::default(),
                HostSlice::from_mut_slice(&mut output),
            )
            .unwrap();
        println!("output= {:?}", output);
        // TODO compare to main device (CUDA by default) or verify with goldens
    }

    #[test]
    fn merkle_tree_keccak() {
        initialize();
        test_utilities::test_set_ref_device();

        // Need a &[&Hashers] to build a tree. Can build it like this
        {
            // build a simple tree with 2 layers
            let leaf_element_size = 8;
            let hasher_l0 = Keccak256::new(2 * leaf_element_size /*input chunk size*/).unwrap();
            let hasher_l1 = Sha3_256::new(2 * 32 /*input chunk size*/).unwrap();
            let layer_hashes = [&hasher_l0, &hasher_l1];
            let _merkle_tree = MerkleTree::new(&layer_hashes[..], leaf_element_size as u64, 0).unwrap();
        }

        // or any way that ends up with &[&Hashers]
        // building a binray tree, each layer takes 2*32B=64B and hashes to 32B
        let nof_layers = 4;
        let num_elements = 1 << nof_layers;
        let leaf_element_size = 32;
        let hasher = Keccak256::new(2 * leaf_element_size /*input chunk size*/).unwrap();
        let layer_hashes: Vec<&Hasher> = (0..nof_layers)
            .map(|_| &hasher)
            .collect();
        let merkle_tree = MerkleTree::new(&layer_hashes[..], leaf_element_size as u64, 0).unwrap();

        // Create a vector of random bytes efficiently
        let mut input: Vec<u8> = vec![0; leaf_element_size as usize * num_elements];
        rand::thread_rng().fill(&mut input[..]); // Fill the vector with random data
        println!("input = {:?}", input);

        merkle_tree
            .build(HostSlice::from_slice(&input), &MerkleTreeConfig::default())
            .unwrap();

        let merkle_proof: MerkleProof = merkle_tree
            .get_proof(HostSlice::from_slice(&input), 1, &MerkleTreeConfig::default())
            .unwrap();
        let root = merkle_proof.get_root::<u64>();
        let path = merkle_proof.get_path::<u8>();
        let (leaf, leaf_idx) = merkle_proof.get_leaf::<u8>();
        println!("root = {:?}", root);
        println!("path = {:?}", path);
        println!("leaf = {:?}, leaf_idx = {}", leaf, leaf_idx);

        let verification_valid = merkle_tree
            .verify(&merkle_proof)
            .unwrap();
        assert_eq!(verification_valid, true);

        // TODOs :
        // (1) test real backends: CPU + CUDA. Can also compare the proofs to see the root, path and leaf are the same.
        // (2) test different cases of input padding
    }
}
