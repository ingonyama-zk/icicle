# Serialization

## Serialization API Overview

### **Structs**

Serialization and deserialization for Icicle structures are facilitated by the [Serde library](https://serde.rs/), allowing flexibility with the choice of serializer. These processes internally call FFI functions to manage the conversion between Rust and the underlying representations.

##### **Methods:**

- **`serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>`**:
  Serializes the structure into a byte array using the provided serializer.

- **`deserialize<'de, D>(deserializer: D) -> Result<Self, D::Error>`**:
  Deserializes a byte array into an Icicle strucure. This method reconstructs the structure from its serialized form.

## **Example Usage**

Below is an example demonstrating how to serialize and deserialize a `SumcheckProof`.

```rust
use serde::{Serialize, Deserialize};
use icicle_core::sumcheck::SumcheckProof;
use serde_json;

let proof: SumcheckProof = /* obtain or construct a proof */;

// Serialize the proof
let serialized_proof = serde_json::to_string(proof).expect("Failed to serialize proof");

// Deserialize the proof
let deserialized_proof: SumcheckProof = serde_json::from_str(&serialized_proof).expect("Failed to deserialize proof");

```

## **Serializable Structs**

The following structs are currently supported for serialization and deserialization:

- `MerkleProof`
- `SumcheckProof`
- `FriProof`

