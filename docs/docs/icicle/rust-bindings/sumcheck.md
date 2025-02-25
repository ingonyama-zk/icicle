# Sumcheck

## Sumcheck API Overview

### **Structs**

#### `SumcheckTranscriptConfig`
Configuration structure for the SumCheck protocol’s transcript.

##### **Fields:**
- `hash: &Hasher` - Reference to the hashing function used.
- `domain_separator_label: Vec<u8>` - Domain separator label for transcript uniqueness.
- `round_poly_label: Vec<u8>` - Label for the polynomial at each round.
- `round_challenge_label: Vec<u8>` - Label for the challenge at each round.
- `little_endian: bool` - Whether to use little-endian encoding.
- `seed_rng: F` - Random number generator seed.

##### **Methods:**
- **`new(hash, domain_separator_label, round_poly_label, round_challenge_label, little_endian, seed_rng) -> Self`**:
  Constructs a new `SumcheckTranscriptConfig` with explicit parameters.

- **`from_string_labels(hash, domain_separator_label, round_poly_label, round_challenge_label, little_endian, seed_rng) -> Self`**:
  Convenience constructor using string labels.

#### `SumcheckConfig`
General configuration for the SumCheck execution.

##### **Fields:**
- `stream: IcicleStreamHandle` - Stream for asynchronous execution (default: `nullptr`).
- `use_extension_field: bool` - Whether to use an extension field for Fiat-Shamir transformation. Sumcheck currently does not support extension fields, always set to `false` otherwise return an error.
- `batch: u64` - Number of input chunks to hash in batch (default: 1).
- `are_inputs_on_device: bool` - Whether inputs reside on the device (e.g., GPU).
- `is_async: bool` - Whether hashing is run asynchronously.
- `ext: ConfigExtension` - Pointer to backend-specific configuration extensions.

##### **Methods:**
- **`default() -> Self`**: 
  Returns a default `SumcheckConfig` instance.

### **Traits**

#### `Sumcheck`
Defines the main API for SumCheck operations.

##### **Associated Types:**
- `Field: FieldImpl + Arithmetic` - The field implementation used.
- `FieldConfig: FieldConfig + GenerateRandom<Self::Field> + FieldArithmetic<Self::Field>` - Field configuration.
- `Proof: SumcheckProofOps<Self::Field>` - Type representing the proof.

##### **Methods:**
- **`new() -> Result<Self, eIcicleError>`**:
  Initializes a new instance.

- **`prove(mle_polys, mle_poly_size, claimed_sum, combine_function, transcript_config, sumcheck_config) -> Self::Proof`**:
  Generates a proof for the polynomial sum over the Boolean hypercube.

- **`verify(proof, claimed_sum, transcript_config) -> Result<bool, eIcicleError>`**:
  Verifies the provided proof.


#### `SumcheckProofOps`
Operations for handling SumCheck proofs.

##### **Methods:**
- **`get_round_polys(&self) -> Result<Vec<Vec<F>>, eIcicleError>`**:
  Retrieves the polynomials for each round.

- **`print(&self) -> eIcicleError`**::
  Prints the proof.


## **Usage Example**

Below is an example demonstrating how to use the `sumcheck` module, adapted from the `check_sumcheck_simple` test.

```rust
use icicle_core::sumcheck::{Sumcheck, SumcheckConfig, SumcheckTranscriptConfig};
use icicle_core::field::FieldElement;
use icicle_core::polynomial::Polynomial;
use icicle_hash::keccak::Keccak256;

fn main() {
    // Initialize hashing function
    let hash = Keccak256::new(0).unwrap();

    // Define a polynomial, e.g., f(x, y) = x + y
    let coefficients = vec![
        FieldElement::from(0), // Constant term
        FieldElement::from(1), // Coefficient for x
        FieldElement::from(1), // Coefficient for y
    ];
    let poly = Polynomial::new(coefficients);

    // Generate mle polynomial
    let mut mle_poly = Vec::with_capacity(2);
    for _ in 0..4 {
        mle_poly.push(poly);
    }

    // Calculate the expected sum over the Boolean hypercube {0,1}^2
    let expected_sum = FieldElement::from(4);

    // Configure transcript and execution settings
    let transcript_config = SumcheckTranscriptConfig::from_string_labels(
        &hash,
        "domain_separator",
        "round_poly",
        "round_challenge",
        false, // big endian
        FieldElement::from(0),
    );
    let sumcheck_config = SumcheckConfig::default();
   
    // define sumcheck lambda
    let combine_func = P::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();
    
    // Initialize prover
    let prover = Sumcheck::new().expect("Failed to create Sumcheck instance");

    // Generate proof
    let proof = prover.prove(
        mle_poly.as_slice(),
        2, // Number of variables in the polynomial
        expected_sum,
        combine_func, // Use pre-defined combine function eq * (a * b - c)
        &transcript_config,
        &sumcheck_config,
    );

    // Verify the proof
    let result = prover.verify(&proof, expected_sum, &transcript_config);
    assert!(result.is_ok() && result.unwrap(), "SumCheck proof verification failed!");
}
