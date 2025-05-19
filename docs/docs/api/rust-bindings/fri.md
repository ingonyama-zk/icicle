# FRI

## FRI API Overview

### **Structs**

#### `FriTranscriptConfig`
Configuration structure for the FRI protocol's transcript.

##### **Fields:**
- `hash: &Hasher` - Reference to the hashing function used.
- `domain_separator_label: String` - Domain separator label for transcript uniqueness.
- `round_challenge_label: String` - Label for the challenge at each round.
- `commit_phase_label: String` - Label for the commit phase.
- `nonce_label: String` - Label for the nonce.
- `public_state: Vec<u8>` - Public state data.
- `seed_rng: F` - The seed for initializing the RNG.

##### **Methods:**
- **`new(hash, domain_separator_label, round_challenge_label, commit_phase_label, nonce_label, public_state, seed_rng) -> Self`**:
  Constructs a new `FriTranscriptConfig` with explicit parameters.

- **`new_default_labels(hash, seed_rng) -> Self`**:
  Constructs a `FriTranscriptConfig` with default labels.

#### `FFIFriTranscriptConfig`
FFI configuration structure for the FRI protocol's transcript.

- Conversion from `FriTranscriptConfig` using `From` trait.

#### `FriProof`
A structure representing the FRI proof, which includes methods for handling proof data.

##### **Methods:**
- **`new() -> Result<Self, eIcicleError>`**:
  Constructs a new instance of the `FriProof`.

- **`create_with_arguments(query_proofs_data, final_poly, pow_nonce) -> Result<Self, eIcicleError>`**:
  Creates a new instance of `FriProof` with the given proof data.

- **`get_query_proofs(&self) -> Result<Vec<Vec<MerkleProofData<F>>>, eIcicleError>`**:
  Returns the matrix of Merkle proofs, where each row corresponds to a query and each column corresponds to a round.

- **`get_final_poly(&self) -> Result<Vec<F>, eIcicleError>`**:
  Returns the final polynomial values.

- **`get_pow_nonce(&self) -> Result<u64, eIcicleError>`**:
  Returns the proof-of-work nonce.

#### `FriConfig`
Configuration structure for the FRI protocol.

##### **Fields:**
- `stream_handle: IcicleStreamHandle` - Stream for asynchronous execution.
- `folding_factor: u64` - The factor by which the codeword is folded in each round.
- `stopping_degree: u64` - The minimal polynomial degree at which folding stops.
- `pow_bits: u64` - Number of leading zeros required for proof-of-work.
- `nof_queries: u64` - Number of queries, computed for each folded layer of FRI.
- `are_inputs_on_device: bool` - True if inputs reside on the device (e.g., GPU).
- `is_async: bool` - True to run operations asynchronously.
- `ext: ConfigExtension` - Pointer to backend-specific configuration extensions.

##### **Methods:**
- **`default() -> Self`**:
  Returns a default `FriConfig` instance with standard settings.

## **Example Usage**

Below is an example demonstrating how to use the `fri` module:

```rust
let merkle_tree_leaves_hash = Keccak256::new(std::mem::size_of::<ScalarField>() as u64).unwrap();
let merkle_tree_compress_hash = Keccak256::new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
let transcript_hash = Keccak256::new(0).unwrap();

const SIZE: u64 = 1 << 10;

init_domain::<ScalarField>(SIZE, false);

let fri_config = FriConfig::default();
let scalars = ScalarCfg::generate_random(SIZE as usize);
let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, ScalarField::one());
let merkle_tree_min_layer_to_store = 0;

let fri_proof = fri_merkle_tree_prove::<ScalarField>(
    &fri_config,
    &transcript_config,
    HostSlice::from_slice(&scalars),
    &merkle_tree_leaves_hash,
    &merkle_tree_compress_hash,
    merkle_tree_min_layer_to_store,
)
.unwrap();

let valid = fri_merkle_tree_verify::<ScalarField>(
    &fri_config,
    &transcript_config,
    &fri_proof,
    &merkle_tree_leaves_hash,
    &merkle_tree_compress_hash,
)
.unwrap();

assert!(valid);
```

## **Links**

For more information on FRI concepts, see [FRI Primitives](../../cpp/fri).
