# FRI API Documentation

## Overview
The Fast Reed-Solomon Interactive Oracle Proof of Proximity (FRI) protocol is used to efficiently verify that a given polynomial has a bounded degree. 

 The Prover asserts that they know a low-degree polynomial F(x) of degree d, and they provide oracle access to a Reed-Solomon (RS) codeword representing evaluations of this polynomial over a domain L:

$$
RS(F(x), L, n) = \{F(1), F(\alpha), F(\alpha^2), ..., F(\alpha^{n-1})\}
$$

where α is a primitive root of unity, and $n = 2^l$ (for $l ∈ Z$) is the domain size.

## How it works
The proof construction consists of three phases: the Commit and Fold Phase, the Proof of Work Phase (optional), and the Query Phase.
Using a Fiat-Shamir (FS) scheme, the proof is generated in a non-interactive manner, enabling the Prover to generate the entire proof and send it to the Verifier for validation.
The polynomial size must be a power of 2 and is passed to the protocol in evaluation form.

### Prover

#### Commit and Fold Phase
* The prover commits to the polynomial evaluations by constructing a Merkle tree.
* A folding step is performed iteratively to reduce the polynomial degree.
* In each step, the polynomial is rewritten using random coefficients derived from Fiat-Shamir hashing, and a new Merkle tree is built for the reduced polynomial.
* This process continues recursively until the polynomial reaches a minimal length.
* Currently, only a folding factor of 2 is supported.

#### Proof of Work Phase (Optional)
* If enabled, the prover is required to find a nonce such that, when hashed with the final Merkle tree root, the result meets a certain number of leading zero bits.

#### Query Phase
* Using the Fiat-Shamir transform, the prover determines the random query indices based on the previously committed Merkle roots.
* For each sampled index, the prover provides the corresponding Merkle proof, showing that the value is part of the committed Merkle tree.
* The prover returns all required data as the FriProof, which is then verified by the verifier.

### Verifier
* The verifier checks the Merkle proofs to ensure the sampled values were indeed committed to in the commit phase.
* The verifier reconstructs the Fiat-Shamir challenges from the prover's commitments and verifies that the prover followed the protocol honestly.
* The folding relation is checked for each sampled query.
* If all checks pass, the proof is accepted as valid.


## C++ API

### Configuration structs
There are two key configuration structs related to the Fri protocol.

#### FriConfig
The `FriConfig` struct is used to specify parameters for the FRI protocol. It contains the following fields:
- **`stream: icicleStreamHandle`**: The CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
- **`folding_factor: size_t`**: The factor by which the codeword is folded in each round.
- **`stopping_degree: size_t`**: The minimal polynomial degree at which folding stops.
- **`pow_bits: size_t`**: Number of leading zeros required for proof-of-work. If set, the optional proof-of-work phase is executed.
- **`nof_queries: size_t`**: Number of queries computed for each folded layer of FRI.
- **`are_inputs_on_device: bool`**: If true, the input polynomials are stored on the device (e.g., GPU); otherwise, they remain on the host (e.g., CPU).
- **`is_async: bool`**: If true, it runs the hash asynchronously.
- **`ext: ConfigExtension*`**: Backend-specific extensions.

The default values are:
```cpp
// icicle/fri/fri_config.h
struct FriConfig {
  icicleStreamHandle stream = nullptr;
  size_t folding_factor = 2;
  size_t stopping_degree = 0;
  size_t pow_bits = 16;
  size_t nof_queries = 100;
  bool are_inputs_on_device = false;
  bool is_async = false;
  ConfigExtension* ext = nullptr;
};
```
:::note
Currently, only a folding factor of 2 is supported.
:::

#### FriTranscriptConfig
The `FriTranscriptConfig<TypeParam>` class is used to specify parameters for the Fiat-Shamir scheme used by the FRI protocol. It contains the following fields:
- **`hasher: Hash`**: The hash function used to generate randomness for Fiat-Shamir.
- **`domain_separator_label: std::vector<std::byte>`**
- **`round_challenge_label: std::vector<std::byte>`**
- **`commit_phase_label: std::vector<std::byte>`**
- **`nonce_label: std::vector<std::byte>`**
- **`public_state: std::vector<std::byte>`**
- **`seed_rng: TypeParam`**: The seed for initializing the RNG.

:::note
The encoding is little endian.
:::

There are three constructors for `FriTranscriptConfig<TypeParam>`:

* **Default constructor**:
```cpp
// icicle/fri/fri_transcript_config.h
FriTranscriptConfig()
  : m_hasher(create_keccak_256_hash()), m_domain_separator_label({}), m_commit_phase_label({}), m_nonce_label({}),
    m_public({}), m_seed_rng(F::zero())
```

* **Constructor with byte vector for labels**:
```cpp
FriTranscriptConfig(
  Hash hasher,
  std::vector<std::byte>&& domain_separator_label,
  std::vector<std::byte>&& round_challenge_label,
  std::vector<std::byte>&& commit_phase_label,
  std::vector<std::byte>&& nonce_label,
  std::vector<std::byte>&& public_state,
  F seed_rng)
    : m_hasher(std::move(hasher)), m_domain_separator_label(std::move(domain_separator_label)),
      m_round_challenge_label(std::move(round_challenge_label)),
      m_commit_phase_label(std::move(commit_phase_label)), m_nonce_label(std::move(nonce_label)),
      m_public(std::move(public_state)), m_seed_rng(seed_rng)
```

* **Constructor with `const char*` arguments for labels**:
```cpp
    FriTranscriptConfig(
      Hash hasher,
      const char* domain_separator_label,
      const char* round_challenge_label,
      const char* commit_phase_label,
      const char* nonce_label,
      std::vector<std::byte>&& public_state,
      F seed_rng)
        : m_hasher(std::move(hasher)), m_domain_separator_label(cstr_to_bytes(domain_separator_label)),
          m_round_challenge_label(cstr_to_bytes(round_challenge_label)),
          m_commit_phase_label(cstr_to_bytes(commit_phase_label)), m_nonce_label(cstr_to_bytes(nonce_label)),
          m_public(std::move(public_state)), m_seed_rng(seed_rng)
```

### Generating FRI Proofs
To generate a proof, first, an empty proof needs to be created. The FRI proof is represented by the `FriProof<TypeParam>` class:

```cpp
// icicle/fri/fri_proof.h
template <typename F>
class FriProof
```

The class has a default constructor `FriProof()` that takes no arguments.

To generate a FRI proof using the Merkle Tree commit scheme, use one of the following functions:
1. **Directly call `prove_fri_merkle_tree`:**
   ```cpp
   template <typename F>
   eIcicleError prove_fri_merkle_tree(
       const FriConfig& fri_config,
       const FriTranscriptConfig<F>& fri_transcript_config,
       const F* input_data,
       const size_t input_size,
       Hash merkle_tree_leaves_hash,
       Hash merkle_tree_compress_hash,
       const uint64_t output_store_min_layer,
       FriProof<F>& fri_proof /* OUT */);
   ```
2. **Use the `fri_merkle_tree` namespace, which internally calls `prove_fri_merkle_tree`:**
   ```cpp
   fri_merkle_tree::prove<TypeParam>( ... );
   ```
   This approach calls `prove_fri_merkle_tree` internally but provides a more structured way to access it.

- **`input_data: const F*`**: Evaluations of The input polynomial.
- **`fri_proof: FriProof<F>&`**: The output `FriProof` object containing the generated proof.
* `merkle_tree_leaves_hash`, `merkle_tree_compress_hash` and `output_store_min_layer` refer to the hashes used in the Merkle Trees built in each round of the folding. For further information about ICICLE's Merkle Trees, see [Merkle-Tree documentation](api/cpp/merkle.md) and [Hash documentation](api/cpp/hash.md).

:::note
`folding_factor` must be divisible by `merkle_tree_compress_hash`.
:::


:::note
An NTT domain is used for proof generation, so before generating a proof, an NTT domain of at least the input_data size must be initialized. For more information see [NTT documentation](api/cpp/ntt.md).
:::

```cpp
NTTInitDomainConfig init_domain_config = default_ntt_init_domain_config();
ntt_init_domain(scalar_t::omega(log_input_size), init_domain_config)
```
:::

#### Example: Generating a Proof

```cpp
// Initialize ntt domain
NTTInitDomainConfig init_domain_config = default_ntt_init_domain_config();
ntt_init_domain(scalar_t::omega(log_input_size), init_domain_config);

// Define hashers for merkle tree
uint64_t merkle_tree_arity = 2;
Hash hash = Keccak256::create(sizeof(TypeParam));                          // hash element -> 32B
Hash compress = Keccak256::create(merkle_tree_arity * hash.output_size()); // hash every 64B to 32B

// set transcript config
const char* domain_separator_label = "domain_separator_label";
const char* round_challenge_label = "round_challenge_label";
const char* commit_phase_label = "commit_phase_label";
const char* nonce_label = "nonce_label";
std::vector<std::byte>&& public_state = {};
TypeParam seed_rng = TypeParam::one();

FriTranscriptConfig<TypeParam> transcript_config(
  hash, domain_separator_label, round_challenge_label, commit_phase_label, nonce_label, std::move(public_state),
  seed_rng);

// set fri config
FriConfig fri_config;
fri_config.nof_queries = 100;
fri_config.pow_bits = 16;
fri_config.folding_factor = 2;
fri_config.stopping_degree = 0;

FriProof<TypeParam> fri_proof;

// get fri proof
eIcicleError err = fri_merkle_tree::prove<TypeParam>(
  fri_config, transcript_config, scalars.get(), input_size, hash, compress, output_store_min_layer, fri_proof);
ICICLE_CHECK(err);

// Release ntt domain
ntt_release_domain<scalar_t>();
```

### Verifying Fri Proofs

To verify the FRI proof using the Merkle Tree commit scheme, use one of the following functions:

1. **Directly call `verify_fri_merkle_tree`**:
```cpp
// icicle/fri/fri.h
template <typename F>
eIcicleError verify_fri_merkle_tree(
    const FriConfig& fri_config,
    const FriTranscriptConfig<F>& fri_transcript_config,
    FriProof<F>& fri_proof,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    bool& valid /* OUT */);
```

2. **Use the `fri_merkle_tree` namespac, which internally calls `verify_fri_merkle_tree`:**
  ```cpp
  fri_merkle_tree::verify<TypeParam>( ... );
  ```

:::note
`FriConfig` and `FriTranscriptConfig` used for generating the proof must be identical to the one used for verification.
:::

#### Example: Verifying a Proof
```cpp
bool valid = false;
eIcicleError err = fri_merkle_tree::verify<TypeParam>(
  fri_config, transcript_config, fri_proof, hash, compress, valid);
ICICLE_CHECK(err);
ASSERT_EQ(true, valid); // Ensure proof verification succeeds
```

After calling `fri_merkle_tree::verify`, the variable `valid` will be set to `true` if the proof is valid, and `false` otherwise.