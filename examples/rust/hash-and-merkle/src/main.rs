use clap::Parser;
use hex;
use icicle_babybear::field::{ScalarCfg as BabybearCfg, ScalarField as BabybearField};
use icicle_core::{
    hash::{HashConfig, Hasher},
    merkle::{MerkleTree, MerkleTreeConfig, PaddingPolicy},
    traits::GenerateRandom,
};
use icicle_hash::{blake2s::Blake2s, keccak::Keccak256};
use icicle_runtime::memory::HostSlice;
use std::time::Instant;

/// Command-line argument parser
#[derive(Parser, Debug)]
struct Args {
    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,
}

/// Load backend and set the device based on input arguments
fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device to {}", args.device_type);

    // Create and set the device
    let device = icicle_runtime::Device::new(&args.device_type, 0); // device_id = 0
    icicle_runtime::set_device(&device).unwrap();
}

/// Example of Keccak-256 hashing using the ICICLE framework
fn keccak_hash_example() {
    // 1. Create a Keccak-256 hasher instance
    let keccak_hasher = Keccak256::new(0 /*=default input size */).unwrap();
    // Note: the default input size is useful in some cases. Can be ignored in this example.

    // 2. Hash a simple string
    let input_str = "I like ICICLE! it's so fast and easy";
    let expected_hash = "9fac4e3dc249b59cc57bdec04c132073f0f6ef3a216ef5b3b75815292fb7a45e";
    let mut output = vec![0u8; 32]; // Output buffer for the hash

    keccak_hasher
        .hash(
            HostSlice::from_slice(input_str.as_bytes()),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();

    // Convert the output to a hex string and print the result
    let output_as_hex_str = hex::encode(output);
    println!("Hash(`{}`) = {:?}", input_str, &output_as_hex_str);
    assert_eq!(expected_hash, output_as_hex_str);

    // 3. Hash field elements (Babybear field elements)
    let input_field_elements = BabybearCfg::generate_random(128); // Generate random field elements
    let mut output = vec![0u8; 32]; // Output buffer for the hash

    keccak_hasher
        .hash(
            HostSlice::from_slice(&input_field_elements),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();

    // Convert the output to a hex string and print the result
    let output_as_hex_str = hex::encode(output);
    println!("Hash(```babybear field elements```) = {:?}", &output_as_hex_str);

    // 4. Hash field elements in batch
    let batch = 1 << 10; // 1024
    let single_hash_nof_elements = 1 << 12; // 4096
    let total_input_elements = batch * single_hash_nof_elements;
    let input_field_elements_batch = BabybearCfg::generate_random(total_input_elements);

    // The size of the output determines the batch size for the hash function
    let mut output = vec![0u8; 32 * batch]; // Output buffer for batch hashing. Doesn't have to be u8.

    let start = Instant::now(); // Start timer for performance measurement
    keccak_hasher
        .hash(
            HostSlice::from_slice(&input_field_elements_batch),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();

    // Print the time taken for hashing in milliseconds
    println!(
        "Hashing {} batches of {} field elements took: {} ms",
        batch,
        single_hash_nof_elements,
        start
            .elapsed()
            .as_millis()
    );

    // NOTE: like other ICICLE apis, this also works with DeviceSlice for on-device data.
}

fn merkle_tree_example() {
    // ICICLE provides the following capabilities for working with Merkle trees:
    //
    // 1. Define the Merkle Tree Structure:
    //    - The first step in constructing a Merkle tree is defining its structure, including the number of layers and the hash function(s) to use at each layer.
    //    - You can apply different hash functions at different layers, giving flexibility in customizing the tree based on your requirements.
    //
    // Let's define a Merkle tree where:
    // - The first layer hashes chunks of 1KB using Keccak-256.
    // - The subsequent layers use a tree with arity=4 (4 inputs per node), meaning each hash will operate on 128B of input (4x32B). For these layers, we will use the Blake2s hash function.

    let leaf_size = 1024; // 1KB for the first layer (each leaf)
    let keccak_hasher = Keccak256::new(leaf_size /* input chunk size */).unwrap();
    let blake2s_hasher = Blake2s::new(128 /* 128B for each node in later layers */).unwrap();

    // Define the hashers for each layer in the tree. For simplicity, we're using Blake2s for the internal layers.
    let _layer_hashes = [
        &keccak_hasher,  // Layer 0: Each leaf is 1KB, hashed to 32B using Keccak256
        &blake2s_hasher, // Layer 1: Hashes 128B chunks to 32B using Blake2s
        &blake2s_hasher, // Layer 2: Continues hashing 128B to 32B
        &blake2s_hasher, // Layer 3: Hashes 128B to 32B, producing the root
    ];

    // Alternatively, the tree structure can be built dynamically using an iterator
    let nof_layers = 4; // Assume this is computed from the input size
    let layer_hashes: Vec<&Hasher> = std::iter::once(&keccak_hasher)
        .chain(std::iter::repeat(&blake2s_hasher).take(nof_layers - 1))
        .collect();

    // Initialize the Merkle tree structure with the specified layers and hash functions.
    let merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0 /* store full tree */).unwrap();
    // NOTE: This tree is defined for a 64KB input size and may need adjustments for larger input sizes. Smaller size is padded.

    // 2. Merkle Tree Construction (Commit Phase):
    //    - The commit phase involves constructing the Merkle tree and computing the root hash, which serves as a cryptographic commitment to the input data.
    //    - You can construct the entire tree or just partial layers if the tree is too large.
    //
    // Generate random input data to commit. We are generating 16,384 Babybear elements (each 4 bytes), for a total of 64KB.
    // This is a synthetic example where a leaf is 256 elements, just to demonstrate the flexibility.
    let leafs = BabybearCfg::generate_random(1 << 14); // 16K Babybear elements (input layer = 64KB)

    // Configure the Merkle tree settings. In this example, zero-padding is used for smaller inputs.
    let mut config = MerkleTreeConfig::default();
    config.padding_policy = PaddingPolicy::ZeroPadding; // Add padding if the input is smaller than expected.

    // Build (commit to) the Merkle tree with the input data.
    merkle_tree
        .build(HostSlice::from_slice(&leafs), &config)
        .unwrap();

    // Retrieve the root commitment. In this example, 256 Babybear field elements make up a single leaf.
    let commitment: &[BabybearField] = merkle_tree
        .get_root()
        .unwrap();
    assert_eq!(commitment.len(), leaf_size as usize / 4); // Ensure the root matches the expected size.

    // 3. Proof of Inclusion (Merkle Proofs):
    //    - After building the Merkle tree, you can generate Merkle proofs to prove the inclusion of specific leaves in the tree.
    //    - A Merkle proof consists of a set of sibling hashes that allow the verifier to reconstruct the root from the leaf.
    //    - Proofs can be pruned to include only necessary hashes or contain all sibling nodes.
    //
    // Generate a Merkle proof for the leaf at index 7.
    let merkle_proof = merkle_tree
        .get_proof(HostSlice::from_slice(&leafs), 7, &config)
        .unwrap();

    // TODO: Consider renaming `get_proof()` to `prove()` for clarity.

    // 4. Serialization:
    //    - ICICLE allows you to generate Merkle proofs, but serialization (e.g., converting the proof into a format for transmission) is left to the user.
    //
    // Serialization of the proof is necessary if it is to be sent or stored for later verification.

    let _pruned = merkle_proof.is_pruned(); // Check if the proof is pruned.
    let _path: &[u8] = merkle_proof.get_path(); // Get the path for inclusion verification.
    let _root: &[u8] = merkle_proof.get_root(); // Get the root associated with the proof.
    let (_leaf, _leaf_idx): (&[u8], u64) = merkle_proof.get_leaf(); // Get the leaf and its index in the tree.

    // Can serialize any part of the proof to any format. You can also use other types instead of u8.
    // For example, if using Poseidon, it would make sense to retrieve path, root, and leaf as field elements.

    // 5. Verification:
    //    - Verifying a Merkle proof involves checking that the provided proof hashes back to the root, thus confirming the inclusion of the data.
    //    - ICICLE provides a `verify` function that checks the validity of the Merkle proof.
    //
    // Verify the proof to check if the leaf is correctly included in the tree.
    let proof_is_valid = merkle_tree
        .verify(&merkle_proof)
        .unwrap();

    // Assert that the verification is successful.
    assert!(proof_is_valid);
    println!("Merkle proof verified successfully!");
}

fn main() {
    // Parse command-line arguments
    let args = Args::parse();
    println!("{:?}", args);

    // Load backend and set the device
    try_load_and_set_backend_device(&args);

    // Execute the Keccak hashing example
    keccak_hash_example();

    // Execute the Merkle-tree example
    merkle_tree_example();
}
