#[cfg(test)]
mod tests {

    use crate::{keccak, sha3};
    use icicle_core::{
        hash::HashConfig,
        merkle::{MerkleProof, MerkleTree, MerkleTreeConfig},
        test_utilities,
    };
    use icicle_runtime::memory::HostSlice;
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

        let keccak_hasher = keccak::create_keccak_256_hasher(0).unwrap();
        let input = vec![0 as u8; 90];
        let mut output = vec![0 as u8; 96]; // 256b * batch
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
    fn sha3_hashing() {
        initialize();
        test_utilities::test_set_ref_device();

        let sha3_hasher = sha3::create_sha3_512_hasher(0).unwrap();
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

        let hasher_l0 = keccak::create_keccak_256_hasher(0).unwrap();
        let hasher_l1 = sha3::create_sha3_256_hasher(0).unwrap();
        let layer_hashes = [hasher_l0, hasher_l1];

        // build a simple tree with 2 layers, hashing 4 elements of 128B to a 256B root
        let leaf_element_size = 128 as u64;
        let merkle_tree = MerkleTree::new(&layer_hashes, leaf_element_size, 0).unwrap();
        let input = vec![0 as u8; leaf_element_size as usize * 4];
        merkle_tree
            .build(HostSlice::from_slice(&input), &MerkleTreeConfig::default())
            .unwrap();

        let merkle_proof: MerkleProof = merkle_tree
            .get_proof(HostSlice::from_slice(&input), 1, &MerkleTreeConfig::default())
            .unwrap();
        let root: &[u8] = merkle_proof.get_root();
        let path: &[u8] = merkle_proof.get_path();
        let (leaf, leaf_idx) = merkle_proof.get_leaf::<u8>();
        println!("root = {:?}", root);
        println!("path = {:?}", path);
        println!("leaf = {:?}, leaf_idx = {}", leaf, leaf_idx);

        // TODO real test
    }
}
