use clap::Parser;
use hex;
use icicle_babybear::field::ScalarCfg as BabybearCfg;
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
    // In this example we will demonstrate how to build a binary Merkle-tree as a string commitment, then prove and verify parts of the string.
    let input_string = String::from(
        "Hello, this is an ICICLE example to commit to a string and open specific parts +Add optional Pad",
    );
    println!("Commiting to the input string (leaf-size=6 bytes): `{}`", &input_string);

    let leaf_size = 6; // Leaf is the value that is opened. We will open 6 chars (bytes) of the string in this example.
    let nof_leafs = input_string
        .chars()
        .count() as u64
        / leaf_size;

    // 1. Define the Merkle Tree Structure:
    //    - The first step in constructing a Merkle-tree is defining its structure, including the number of layers and the hash function(s) to use at each layer.
    //    - You can apply different hash functions at different layers, giving flexibility in customizing the tree based on your requirements.
    //
    // Let's define a binary Merkle tree using Keccak-256 to hash the leaves, then Blake2s for the remaining layers
    // Note: typical use case is using up to two different hash functions (usually one)

    let hasher: Hasher = Keccak256::new(leaf_size /*=default input size */).unwrap(); // will hash every 4 bytes
    let compress: Hasher = Blake2s::new(hasher.output_size() * 2 /*=default input size */).unwrap(); // compress every two child nodes to one using blake2s

    let tree_height = nof_leafs.ilog2() as usize;
    let layer_hashes: Vec<&Hasher> = std::iter::once(&hasher)
        .chain(std::iter::repeat(&compress).take(tree_height))
        .collect();

    let merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0 /* min tree layer to store */).unwrap();
    // Note that we can store only the top layers of the tree if it is very large for the device memory.

    // 2. Merkle Tree Construction (Commit Phase):
    //    - The commit phase involves constructing the Merkle tree and computing the root hash, which serves as a cryptographic commitment to the input data.
    //    - You can construct and store the entire tree or just partial layers if the tree is too large.

    // Configure the Merkle tree settings. In this example, zero-padding is used for when input is too small for the tree structure.
    let mut config = MerkleTreeConfig::default();
    config.padding_policy = PaddingPolicy::ZeroPadding; // Add padding if the input is smaller than expected.
                                                        // TODO: padding not supported in v3.1. Expected in v3.2

    // Build (commit to) the Merkle tree with the input data.
    let input = input_string.as_bytes();
    merkle_tree
        .build(HostSlice::from_slice(&input), &config)
        .unwrap();

    // Retrieve the root commitment
    let commitment: &[u8] = merkle_tree
        .get_root()
        .unwrap();
    println!("Tree.root         = 0x{}", hex::encode(commitment));

    // 3. Proof of Inclusion (Merkle Proofs):
    //    - After building the Merkle tree, you can generate Merkle proofs to prove the inclusion of specific leaves in the tree.
    //    - A Merkle proof consists of a set of sibling hashes that allow the verifier to reconstruct the root from the leaf.
    //    - Proofs can be pruned to include only necessary hashes or contain all sibling nodes.

    // Generate a Merkle proof for the leaf at index 3.

    let merkle_proof = merkle_tree
        .get_proof(HostSlice::from_slice(&input), 3, true /*=pruned*/, &config)
        .unwrap();

    // 4. Serialization:
    //    - ICICLE allows you to generate Merkle proofs, but serialization (e.g., converting the proof into a format for transmission) is left to the user.

    let (leaf, opened_leaf_idx): (&[u8], u64) = merkle_proof.get_leaf(); // Get the leaf and its index in the tree.
    let merkle_path: &[u8] = merkle_proof.get_path();

    println!("Proof.pruned      = {}", merkle_proof.is_pruned());
    println!("Proof.commitment  = 0x{}", hex::encode(&merkle_proof.get_root()));
    println!("Proof.leaf_idx    = {}", opened_leaf_idx);
    println!("Proof.leaf        = '{}'", std::str::from_utf8(leaf).unwrap());
    let layer_hash_len = 32; // Note that this print loop assumes that the path is 32B per layer. This it not always the case but to simplify the example.
    for (i, chunk) in merkle_path
        .chunks(layer_hash_len)
        .enumerate()
    {
        println!("Proof.path.layer{}    = {}", i, hex::encode(chunk));
    }
    // Note that you can get various types of slices from the merkle-proof object. // For example, if using Poseidon, it would make sense to take a &[FieldElementType]

    // 5. Verification:
    //    - Verifying a Merkle proof involves checking that the provided proof hashes back to the root, thus confirming the inclusion of the data and its location in the input.
    //    - ICICLE provides a `verify` function that checks the validity of the Merkle proof. Note that the verify part is fully open sourced and implemented only for CPU.

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
    // keccak_hash_example();

    // Execute the Merkle-tree example
    // TODO remove this when merkle-tree works on CUDA backend
    println!("\nWARNING: merkle-tree example falling back to CPU");
    icicle_runtime::set_device(&icicle_runtime::Device::new("CPU", 0)).unwrap();
    merkle_tree_example();
}
