
# Merkle Tree API Documentation (Rust)

This is the Rust version of the **Merkle Tree API Documentation** ([C++ documentation](../primitives/merkle.md)). It mirrors the structure and functionality of the C++ version, providing equivalent APIs in Rust.

---

## What is a Merkle Tree?

A **Merkle tree** is a cryptographic data structure that allows for **efficient verification of data integrity**. It consists of:
- **Leaf nodes**, each containing a piece of data.
- **Internal nodes**, which store the **hashes of their child nodes**, leading up to the **root node** (the cryptographic commitment).

---

## Tree Structure and Configuration in Rust

### Defining a Merkle Tree

With ICICLE, you have the **flexibility** to build various tree topologies based on your needs. A tree is defined by:

1. **Hasher per layer** ([Link to Hasher API](../rust-bindings/hash.md)) with a **default input size**.
2. **Size of a leaf element** (in bytes): This defines the **granularity** of the data used for opening proofs.

The **root node** is assumed to be a single node. The **height of the tree** is determined by the **number of layers**.
Each layer's **arity** is calculated as:

$$
{arity}_i = \frac{layers[i].inputSize}{layer[i-1].outputSize}
$$

For **layer 0**:

$$
{arity}_0 = \frac{layers[0].inputSize}{leafSize}
$$

:::note
Each layer has a shrinking-factor defined by $\frac{layer.outputSize}{layer.inputSize}$.
This factor is used to compute the input size, assuming a single root node.
:::


```rust
struct MerkleTree{
    /// * `layer_hashes` - A vector of hash objects representing the hashes of each layer.
    /// * `leaf_element_size` - Size of each leaf element.
    /// * `output_store_min_layer` - Minimum layer at which the output is stored.
    ///
    /// # Returns a new `MerkleTree` instance or eIcicleError.
    pub fn new(
        layer_hashes: &[&Hasher],
        leaf_element_size: u64,
        output_store_min_layer: u64,
    ) -> Result<Self, eIcicleError>;
}
```

---

### Building the Tree

The Merkle tree can be constructed from input data of any type, allowing flexibility in its usage. The size of the input must align with the tree structure defined by the hash layers and leaf size. If the input size does not match the expected size, padding may be applied.

Refer to the Padding Section for more details on how mismatched input sizes are handled.

```rust
struct MerkleTree{
    /// * `leaves` - A slice of leaves (input data).
    /// * `config` - Configuration for the Merkle tree.
    ///
    /// # Returns a result indicating success or failure.
    pub fn build<T>(
        &self,
        leaves: &(impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &MerkleTreeConfig,
    ) -> Result<(), eIcicleError>;
}
```

---

## Tree Examples in Rust

### Example A: Binary Tree

A binary tree with **5 layers**, using **Keccak-256**:

![Merkle Tree Diagram](../primitives/merkle_diagrams/diagram1.png)

```rust
use icicle_core::{
    hash::{HashConfig, Hasher},
    merkle::{MerkleTree, MerkleTreeConfig},
};
use icicle_hash::keccak::Keccak256;
use icicle_runtime::memory::HostSlice;

let leaf_size = 1024_u64;
let max_input_size = leaf_size as usize * 16;
let input: Vec<u8> = vec![0; max_input_size];

// define layer hashes
// we want one hash layer to hash every 1KB to 32B then compress every 64B so 4 more binary layers
let hash = Keccak256::new(leaf_size).unwrap();
let compress = Keccak256::new(2 * hash.output_size()).unwrap();
let _layer_hashes = vec![&hash, &compress, &compress, &compress, &compress];
// or like that
let layer_hashes: Vec<&Hasher> = std::iter::once(&hash)
    .chain(std::iter::repeat(&compress).take(4))
    .collect();

let merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0 /*min layer to store */).unwrap();

// compute the tree
merkle_tree
    .build(HostSlice::from_slice(&input), &MerkleTreeConfig::default())
    .unwrap();
```

---

### Example B: Tree with Arity 4

![Merkle Tree Diagram](../primitives/merkle_diagrams/diagram2.png)

This example uses **Blake2s** in upper layers:

```rust
use icicle_hash::blake2s::Blake2s;

// define layer hashes
// we want one hash layer to hash every 1KB to 32B then compress every 128B so only 2 more layers
let hash = Keccak256::new(leaf_size).unwrap();
let compress = Blake2s::new(4 * hash.output_size()).unwrap();
let layer_hashes = vec![&hash, &compress, &compress];

let merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 0 /*min layer to store */).unwrap();

merkle_tree
    .build(HostSlice::from_slice(&input), &MerkleTreeConfig::default())
    .unwrap();
```

---

## Padding

:::note
Padding feature is not yet supported in **v3.1** and will be available in **v3.2**.
:::

When the input for **layer 0** is smaller than expected, ICICLE can apply **padding** to align the data.

**Padding Schemes:**
1. **Zero padding:** Adds zeroes to the remaining space.
2. **Repeat last leaf:** The final leaf element is repeated to fill the remaining space.

```rust
// pub enum PaddingPolicy {
//     None,        // No padding, assume input is correctly sized.
//     ZeroPadding, // Pad the input with zeroes to fit the expected input size.
//     LastValue,   // Pad the input by repeating the last value.
// }

use icicle_core::merkle::PaddingPolicy;
let mut config = MerkleTreeConfig::default();
config.padding_policy = PaddingPolicy::ZeroPadding;
merkle_tree
    .build(HostSlice::from_slice(&input), &config)
    .unwrap();
```

---

## Root as Commitment

Retrieve the Merkle-root and serialize.

```rust
struct MerkleTree{
    /// Retrieve the root of the Merkle tree.
    ///
    /// # Returns
    /// A reference to the root hash.
    pub fn get_root<T>(&self) -> Result<&[T], eIcicleError>;
}

let commitment: &[u8] = merkle_tree
        .get_root()
        .unwrap();
println!("Commitment: {:?}", commitment);****
```

:::note
The commitment can be serialized to the proof. This is not handled by ICICLE.
:::

---

## Generating Merkle Proofs

Merkle proofs are used to **prove the integrity of opened leaves** in a Merkle tree. A proof ensures that a specific leaf belongs to the committed data by enabling the verifier to reconstruct the **root hash (commitment)**.

A Merkle proof contains:

- **Leaf**: The data being verified.
- **Index** (leaf_idx): The position of the leaf in the original dataset.
- **Path**: A sequence of sibling hashes (tree nodes) needed to recompute the path from the leaf to the root.

![Merkle Pruned Phat Diagram](../primitives/merkle_diagrams/diagram1_path.png)

```rust
struct MerkleTree{
    /// * `leaves` - A slice of leaves (input data).
    /// * `leaf_idx` - Index of the leaf to generate a proof for.
    /// * `pruned_path` - Whether the proof should be pruned.
    /// * `config` - Configuration for the Merkle tree.
    ///
    /// # Returns a `MerkleProof` object or eIcicleError
    pub fn get_proof<T>(
        &self,
        leaves: &(impl HostOrDeviceSlice<T> + ?Sized),
        leaf_idx: u64,
        pruned_path: bool,
        config: &MerkleTreeConfig,
    ) -> Result<MerkleProof, eIcicleError>;
}
```

### Example: Generating a Proof

Generating a proof for leaf idx 5:

```rust
let merkle_proof = merkle_tree
    .get_proof(
        HostSlice::from_slice(&input),
        5,    /*=leaf-idx*/
        true, /*pruned*/
        &MerkleTreeConfig::default(),
    )
    .unwrap();
```

:::note
The Merkle-path can be serialized to the proof along the leaf. This is not handled by ICICLE.
:::

---

## Verifying Merkle Proofs

```rust
struct MerkleTree{
    /// * `proof` - The Merkle proof to verify.
    ///
    /// # Returns a result indicating whether the proof is valid.
    pub fn verify(&self, proof: &MerkleProof) -> Result<bool, eIcicleError>;
}
```

### Example: Verifying a Proof

```rust
let valid = merkle_tree
    .verify(&merkle_proof)
    .unwrap();
assert!(valid);
```

---

## Pruned vs. Full Merkle-paths

A **Merkle path** is a collection of **sibling hashes** that allows the verifier to **reconstruct the root hash** from a specific leaf.
This enables anyone with the **path and root** to verify that the **leaf** belongs to the committed dataset.
There are two types of paths that can be computed:

- [**Pruned Path:**](#generating-merkle-proofs) Contains only necessary sibling hashes.
- **Full Path:** Contains all sibling nodes and intermediate hashes.

![Merkle Full Path Diagram](../primitives//merkle_diagrams/diagram1_path_full.png)

To compute a full path, specify `pruned=false`:

```rust
let merkle_proof = merkle_tree
    .get_proof(
        HostSlice::from_slice(&input),
        5,    /*=leaf-idx*/
        false, /*non-pruned is a full path --> note the pruned flag here*/
        &MerkleTreeConfig::default(),
    )
    .unwrap();
```

---

## Handling Partial Tree Storage

In cases where the **Merkle tree is large**, only the **top layers** may be stored to conserve memory.
When opening leaves, the **first layers** (closest to the leaves) are **recomputed dynamically**.

For example to avoid storing first layer we can define a tree as follows:


```rust
let mut merkle_tree = MerkleTree::new(&layer_hashes, leaf_size, 1 /*min layer to store*/);
```
