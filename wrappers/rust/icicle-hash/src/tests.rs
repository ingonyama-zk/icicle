#[cfg(test)]
mod tests {

    use crate::{
        blake2s::Blake2s,
        blake3::Blake3,
        keccak::{Keccak256, Keccak512},
        pow::{pow_solver, pow_verify, PowConfig},
        sha3::Sha3_256,
    };
    use icicle_core::{
        hash::{HashConfig, Hasher},
        merkle::{MerkleProof, MerkleTree, MerkleTreeConfig, PaddingPolicy},
    };
    use icicle_runtime::{
        memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut},
        test_utilities,
    };

    use rand::Rng;
    use std::sync::Once;
    use std::time::Instant;

    static INIT: Once = Once::new();

    pub fn initialize() {
        INIT.call_once(move || {
            test_utilities::test_load_and_init_devices();
        });
    }

    #[test]
    fn keccak_hashing_batch() {
        initialize();
        let single_hash_input_size = 30;
        let batch = 3;

        let mut input = vec![0 as u8; single_hash_input_size * batch];
        rand::thread_rng().fill(input.as_mut_slice());
        let mut output_ref = vec![0 as u8; 64 * batch]; // 64B (=512b) is the output size of Keccak512,
        let mut output_main = vec![0 as u8; 64 * batch];

        test_utilities::test_set_ref_device();
        let keccak_hasher = Keccak512::new(0 /*default chunk size */).unwrap();
        keccak_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_ref.into_slice_mut())
            .unwrap();

        test_utilities::test_set_main_device();
        let keccak_hasher = Keccak512::new(0 /*default chunk size */).unwrap();
        keccak_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_main.into_slice_mut())
            .unwrap();
        assert_eq!(output_ref, output_main);
    }

    #[test]
    fn blake2s_hashing() {
        initialize();
        let single_hash_input_size = 567;
        let batch = 11;

        let mut input = vec![0 as u8; single_hash_input_size * batch];
        rand::thread_rng().fill(input.as_mut_slice());
        let mut output_ref = vec![0 as u8; 32 * batch]; // 32B (=256b) is the output size of blake2s
        let mut output_main = vec![0 as u8; 32 * batch];

        test_utilities::test_set_ref_device();
        let blake2s_hasher = Blake2s::new(0 /*default chunk size */).unwrap();
        blake2s_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_ref.into_slice_mut())
            .unwrap();

        test_utilities::test_set_main_device();
        let blake2s_hasher = Blake2s::new(0 /*default chunk size */).unwrap();
        blake2s_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_main.into_slice_mut())
            .unwrap();
        assert_eq!(output_ref, output_main);
    }

    #[test]
    fn blake3_hashing_cpu_gpu() {
        initialize();
        let single_hash_input_size = 567;
        let batch = 11;

        let mut input = vec![0 as u8; single_hash_input_size * batch];
        rand::thread_rng().fill(input.as_mut_slice());
        let mut output_ref = vec![0 as u8; 32 * batch]; // 32B (=256b) is the output size of blake3
        let mut output_main = vec![0 as u8; 32 * batch];

        test_utilities::test_set_ref_device();
        let blake3_hasher = Blake3::new(0 /*default chunk size */).unwrap();
        blake3_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_ref.into_slice_mut())
            .unwrap();

        test_utilities::test_set_main_device();
        let blake3_hasher = Blake3::new(0 /*default chunk size */).unwrap();
        blake3_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_main.into_slice_mut())
            .unwrap();
        assert_eq!(output_ref, output_main);
    }

    #[test]
    fn blake3_hashing() {
        // Known input string and expected hash
        let input_string = "Hello world I am blake3. This is a semi-long Rust test with a lot of characters. 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let expected_hash = "ee4941ff90437a4fd7489ffa6d559e644a68b2547e95a690949b902da128b273";

        let input = input_string.as_bytes();
        let mut output_ref = vec![0u8; 32]; // 32B (=256b) is the output size of blake3

        test_utilities::test_set_ref_device();
        let blake3_hasher = Blake3::new(0 /*default chunk size */).unwrap();
        blake3_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_ref.into_slice_mut())
            .unwrap();

        // Convert output_ref to hex for comparison
        let output_ref_hex: String = output_ref
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        assert_eq!(
            output_ref_hex, expected_hash,
            "Hash mismatch: got {}, expected {}",
            output_ref_hex, expected_hash
        );

        println!("Test passed: Computed hash matches expected hash.");
    }

    #[test]
    fn sha3_hashing() {
        initialize();
        let mut input = vec![0 as u8; 1153];
        rand::thread_rng().fill(input.as_mut_slice());
        let mut output_main = vec![0 as u8; 32];
        let mut output_ref = vec![0 as u8; 32];

        test_utilities::test_set_ref_device();
        let sha3_hasher = Sha3_256::new(0 /*default chunk size */).unwrap();
        sha3_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_ref.into_slice_mut())
            .unwrap();

        test_utilities::test_set_main_device();
        let sha3_hasher = Sha3_256::new(0 /*default chunk size */).unwrap();
        sha3_hasher
            .hash(input.into_slice(), &HashConfig::default(), output_main.into_slice_mut())
            .unwrap();

        assert_eq!(output_ref, output_main);
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
            let _merkle_tree = MerkleTree::new(layer_hashes.as_slice(), leaf_element_size as u64, 0).unwrap();
        }

        // or any way that ends up with &[&Hashers]
        // building a binary tree, each layer takes 2*32B=64B and hashes to 32B
        let nof_layers = 4;
        let num_elements = 1 << nof_layers;
        let leaf_element_size = 32;
        let hasher = Keccak256::new(2 * leaf_element_size /*input chunk size*/).unwrap();
        let layer_hashes: Vec<&Hasher> = (0..nof_layers)
            .map(|_| &hasher)
            .collect();
        let merkle_tree = MerkleTree::new(layer_hashes.as_slice(), leaf_element_size as u64, 0).unwrap();

        // Create a vector of random bytes efficiently
        let mut input: Vec<u8> = vec![0; leaf_element_size as usize * num_elements];
        rand::thread_rng().fill(input.as_mut_slice()); // Fill the vector with random data

        merkle_tree
            .build(input.into_slice(), &MerkleTreeConfig::default())
            .unwrap();

        let merkle_proof: MerkleProof = merkle_tree
            .get_proof(
                input.into_slice(),
                1,
                false, /*=pruned*/
                &MerkleTreeConfig::default(),
            )
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

        // Now we will build the same tree on the main device
        test_utilities::test_set_main_device();
        let hasher = Keccak256::new(2 * leaf_element_size /*input chunk size*/).unwrap();
        let layer_hashes: Vec<&Hasher> = (0..nof_layers)
            .map(|_| &hasher)
            .collect();
        let merkle_tree = MerkleTree::new(layer_hashes.as_slice(), leaf_element_size as u64, 0).unwrap();
        merkle_tree
            .build(input.into_slice(), &MerkleTreeConfig::default())
            .unwrap();

        let merkle_proof: MerkleProof = merkle_tree
            .get_proof(
                input.into_slice(),
                1,
                false, /*=pruned*/
                &MerkleTreeConfig::default(),
            )
            .unwrap();
        let main_root = merkle_proof.get_root::<u64>();
        let main_path = merkle_proof.get_path::<u8>();
        let (main_leaf, _leaf_idx) = merkle_proof.get_leaf::<u8>();
        let verification_valid = merkle_tree
            .verify(&merkle_proof)
            .unwrap();
        assert_eq!(verification_valid, true);

        // Check if results for main and ref are equal
        assert_eq!(root, main_root);
        assert_eq!(path, main_path);
        assert_eq!(leaf, main_leaf);

        // test proving merkle-proof with device memory too
        let device_leaves = DeviceVec::from_host_slice(&input);
        let merkle_proof_from_device_mem: MerkleProof = merkle_tree
            .get_proof(&device_leaves, 2, false /*=pruned*/, &MerkleTreeConfig::default())
            .unwrap();
        assert_eq!(
            merkle_tree
                .verify(&merkle_proof_from_device_mem)
                .unwrap(),
            true
        );
    }

    #[test]
    fn blake3_pow() {
        initialize();
        test_utilities::test_set_main_device();
        const BITS: u8 = 25;
        let input: [u8; 32] = [20; 32];
        let golden_nonce: u64 = 40825909;
        let golden_hash: u64 = 364385878471;
        let input_host = input.into_slice();
        let cfg = PowConfig::default();

        let mut gpu_found = false;
        let mut gpu_nonce = 0;
        let mut gpu_mined_hash = 0;

        let hasher = Blake3::new(0).unwrap();

        pow_solver(
            &hasher,
            input_host,
            BITS,
            &cfg,
            &mut gpu_found,
            &mut gpu_nonce,
            &mut gpu_mined_hash,
        )
        .unwrap();
        assert!(gpu_found);
        assert_eq!(gpu_nonce, golden_nonce);
        assert_eq!(gpu_mined_hash, golden_hash);

        let mut gpu_is_correct = false;
        let mut gpu_mined_hash_check = 0;

        pow_verify(
            &hasher,
            input_host,
            BITS,
            &cfg,
            gpu_nonce,
            &mut gpu_is_correct,
            &mut gpu_mined_hash_check,
        )
        .unwrap();
        assert_eq!(gpu_mined_hash_check, golden_hash);
        assert!(gpu_is_correct);

        test_utilities::test_set_ref_device();
        let mut cpu_found = false;
        let mut cpu_nonce = 0;
        let mut cpu_mined_hash = 0;
        let hasher = Blake3::new(0).unwrap();
        pow_solver(
            &hasher,
            input_host,
            BITS,
            &cfg,
            &mut cpu_found,
            &mut cpu_nonce,
            &mut cpu_mined_hash,
        )
        .unwrap();
        assert!(cpu_found);
        assert_eq!(cpu_nonce, golden_nonce);
        assert_eq!(cpu_mined_hash, golden_hash);

        let mut cpu_is_correct = false;
        let mut cpu_mined_hash_check = 0;

        pow_verify(
            &hasher,
            input_host,
            BITS,
            &cfg,
            cpu_nonce,
            &mut cpu_is_correct,
            &mut cpu_mined_hash_check,
        )
        .unwrap();
        assert_eq!(cpu_mined_hash, golden_hash);
        assert!(cpu_is_correct);
    }
    #[test]
    fn keccak_pow() {
        initialize();
        test_utilities::test_set_main_device();
        const BITS: u8 = 25;
        let input: [u8; 21] = [20; 21];

        let input_host = input.into_slice();
        let mut cfg = PowConfig::default();
        cfg.padding_size = 3;

        let mut gpu_found = false;
        let mut gpu_nonce = 0;
        let mut gpu_mined_hash = 0;

        let hasher = Keccak256::new(0).unwrap();

        pow_solver(
            &hasher,
            input_host,
            BITS,
            &cfg,
            &mut gpu_found,
            &mut gpu_nonce,
            &mut gpu_mined_hash,
        )
        .unwrap();
        assert!(gpu_found);

        let mut gpu_is_correct = false;
        let mut gpu_mined_hash_check = 0;

        pow_verify(
            &hasher,
            input_host,
            BITS,
            &cfg,
            gpu_nonce,
            &mut gpu_is_correct,
            &mut gpu_mined_hash_check,
        )
        .unwrap();
        assert_eq!(gpu_mined_hash_check, gpu_mined_hash);
        assert!(gpu_is_correct);

        test_utilities::test_set_ref_device();
        let mut cpu_found = false;
        let mut cpu_nonce = 0;
        let mut cpu_mined_hash = 0;
        let hasher = Keccak256::new(0).unwrap();
        pow_solver(
            &hasher,
            input_host,
            BITS,
            &cfg,
            &mut cpu_found,
            &mut cpu_nonce,
            &mut cpu_mined_hash,
        )
        .unwrap();
        assert!(cpu_found);
        assert_eq!(cpu_nonce, gpu_nonce);
        assert_eq!(cpu_mined_hash, gpu_mined_hash);

        let mut cpu_is_correct = false;
        let mut cpu_mined_hash_check = 0;

        pow_verify(
            &hasher,
            input_host,
            BITS,
            &cfg,
            cpu_nonce,
            &mut cpu_is_correct,
            &mut cpu_mined_hash_check,
        )
        .unwrap();
        assert_eq!(cpu_mined_hash, cpu_mined_hash_check);
        assert!(cpu_is_correct);
    }

    #[test]
    fn test_merkle_tree_segfault() {
        initialize();
        test_utilities::test_set_main_device();

        let n = 18;
        let field_element_size = 4;
        for elements_per_leaf in [1, 5] {
            // calc tree parameters
            let leaf_size = field_element_size * elements_per_leaf;
            let mut test_vec = vec![
                0 as u8;
                (field_element_size * (1 << n))
                    .try_into()
                    .unwrap()
            ];
            rand::thread_rng().fill(&mut test_vec[..]);
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
                    .build(test_vec.into_slice(), &config)
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
