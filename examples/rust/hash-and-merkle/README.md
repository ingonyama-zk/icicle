# ICICLE Example: Hashing and Merkle-Tree

## Key Takeaway

The examples cover:
- **String Hashing** using Keccak-256.
- **Field Element Hashing** with Babybear and Keccak-256.
- **Merkle Tree Construction** using Keccak-256 and Blake2s.

> [!NOTE]
> While Keccak-256 and blake2s are used here, other supported hash functions (e.g., SHA3, Poseidon and others) can be substituted.

## What's in this Example

1. **Backend Setup**: Configure ICICLE to use either the **CPU** or **CUDA** backend.
2. **String Hashing**: Demonstrates hashing strings using Keccak-256.
3. **Field Element Hashing**: Hash Babybear field elements using Keccak-256.
4. **Merkle Tree Construction**: Build, prove, and verify a Merkle tree using Keccak-256 and blake2s.

## 1. Hashing Example

This example hashes both strings and field elements using **Keccak-256**, commonly used in cryptographic systems.

The **batch size** for hashing is automatically derived from the size of the output buffer, meaning it scales according to the size of the output you need. 
This design also allows the same hashing function to efficiently handle any input type, whether it’s simple strings, scalar field elements, or elliptic curve points.

```rust
// Initialize the Keccak-256 hasher
let keccak_hasher = Keccak256::new(0).unwrap();

// Hash a simple string
let input_str = "I like ICICLE! it's so fast and easy";
let mut output = vec![0u8; 2*32]; // Output buffer for batch=2
keccak_hasher.hash(
    HostSlice::from_slice(input_str.as_bytes()),
    &HashConfig::default(),
    HostSlice::from_mut_slice(&mut output),
).unwrap();

// Hash Babybear field elements
let input_field_elements = BabybearCfg::generate_random(128);
let mut output = vec![0u8; 1*32]; // Output buffer for batch=1
keccak_hasher.hash(
    HostSlice::from_slice(&input_field_elements),
    &HashConfig::default(),
    HostSlice::from_mut_slice(&mut output),
).unwrap();
```

### 2. Merkle Tree Example

The Merkle tree example demonstrates **building a binary Merkle tree**, committing to string data, and generating **Merkle proofs**.

```rust
fn merkle_tree_example() {
    let input_string = String::from(
        "Hello, this is an ICICLE example to commit to a string and open specific parts +Add optional Pad",
    );
    let input = input_string.as_bytes();
    
    // define tree with Keccak-256 and blake2s hashers
    let hasher = Keccak256::new(leaf_size).unwrap();
    let compress = Blake2s::new(hasher.output_size() * 2).unwrap();    
    let layer_hashes: Vec<&Hasher> = std::iter::once(&hasher)
        .chain(std::iter::repeat(&compress).take(tree_height))
        .collect();
    let merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0).unwrap();

    // 1. Commit
    merkle_tree.build(HostSlice::from_slice(&input), &config).unwrap();
    let commitment = merkle_tree.get_root().unwrap();
    // 2. Open leaf #3
    let merkle_proof = merkle_tree
        .get_proof(HostSlice::from_slice(&input), 3, true /*=pruned path*/, &config)
        .unwrap();
    // 3. Verify: (Note that we reconstruct the tree here, as verification is typically performed by a separate application)
    let verifier_merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0).unwrap();
    let proof_is_valid = verifier_merkle_tree.verify(&merkle_proof).unwrap();
}
```

This example demonstrates **building and verifying Merkle trees** using the ICICLE framework. The tree uses **Keccak-256** for leaf hashing and **Blake2s** for internal nodes.


## Running the Example

Use the following commands to run the example based on your backend:

```sh
# For CPU
./run.sh -d CPU

# For CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```
