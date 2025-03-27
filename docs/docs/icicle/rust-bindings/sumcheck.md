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
use icicle_bn254::sumcheck::SumcheckWrapper as SW;
use icicle_core::program::{PreDefinedProgram, ReturningValueProgram as P};
use icicle_core::sumcheck::{Sumcheck, SumcheckConfig, SumcheckProofOps, SumcheckTranscriptConfig};
use icicle_core::traits::{FieldImpl, GenerateRandom};
use icicle_hash::keccak::Keccak256;
use icicle_runtime::memory::HostSlice;

//setup
let log_mle_poly_size = 10u64;
let mle_poly_size = 1 << log_mle_poly_size;
//number of MLE polys
let nof_mle_poly = 4;
let mut mle_polys = Vec::with_capacity(nof_mle_poly);
//create polys
for _ in 0..nof_mle_poly {
  let mle_poly_random = <<SW as Sumcheck>::FieldConfig>::generate_random(mle_poly_size);
  mle_polys.push(mle_poly_random);
}
//compute claimed sum
let mut claimed_sum = <<SW as Sumcheck>::Field as FieldImpl>::zero();
for i in 0..mle_poly_size {
  let a = mle_polys[0][i];
  let b = mle_polys[1][i];
  let c = mle_polys[2][i];
  let eq = mle_polys[3][i];
  claimed_sum = claimed_sum + (a * b - c) * eq;
}
//create polynomial host slices
let mle_poly_hosts = mle_polys
    .iter()
    .map(|poly| HostSlice::from_slice(poly))
    .collect::<Vec<&HostSlice<<SW as Sumcheck>::Field>>>();
//define transcript config
let leaf_size:u64 = (<SW as Sumcheck>::Field::one()).to_bytes_le().len().try_into().unwrap();
let hasher = Keccak256::new(0).unwrap();
let seed_rng = <<SW as Sumcheck>::FieldConfig>::generate_random(1)[0];
let transcript_config = SumcheckTranscriptConfig::from_string_labels(
        &hasher,
        "DomainLabel",
        "PolyLabel",
        "ChallengeLabel",
        true, // big endian
        seed_rng,
    );
//define sumcheck config
let sumcheck_config = SumcheckConfig::default();
let sumcheck = <SW as Sumcheck>::new().unwrap();
//define combine function
let combine_function = <icicle_bn254::program::bn254::FieldReturningValueProgram as ReturningValueProgram>::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();
let proof = sumcheck.prove(
        &mle_poly_hosts,
        mle_poly_size.try_into().unwrap(),
        claimed_sum,
        combine_function,
        &transcript_config,
        &sumcheck_config,);
//serialize round polynomials from proof
let proof_round_polys = <<SW as Sumcheck>::Proof as SumcheckProofOps<
        <SW as Sumcheck>::Field,>>::get_round_polys(&proof).unwrap();
//verifier reconstruct proof from round polynomials
let proof_as_sumcheck_proof: <SW as Sumcheck>::Proof =
        <SW as Sumcheck>::Proof::from(proof_round_polys);
//verify proof
let proof_validty = sumcheck.verify(&proof_as_sumcheck_proof, claimed_sum, &transcript_config);
println!("Sumcheck proof verified, is valid: {}", proof_validty.unwrap());
```
# Misc
## ReturningValueProgram
A variant of [Program](./program.md) tailored for Sumcheck's combine function. It differs from `Program` by the function it receives in its constructor - instead of returning no value and using the given parameter vector as both inputs and outputs, it returns a single value which is the one and only return value of the function. This way it fulfils the utility of the combine function, allowing custom combine functions for the icicle backend.
```rust
pub trait ReturningValueProgram:
  Sized + Handle
{
  type Field: FieldImpl;
  type ProgSymbol: Symbol<Self::Field>;

  fn new(program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>) -> Self::ProgSymbol, nof_parameters: u32) -> Result<Self, eIcicleError>;

  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
}
```
