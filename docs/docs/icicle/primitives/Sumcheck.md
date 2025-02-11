# Sumcheck API Documentation

## Overview
Sumchek protocol is a protocol where a Prover prove to a Verifier that the sum of a multilinear polynomial, over the boolean hypercube, is a specific sum.

For a polynomial $P$, with $n$ dimention $n$ the Prover is trying to prove that:
$$
\sum_{X_1 \in \{0, 1\}}\sum_{X_2 \in \{0, 1\}}\cdot\cdot\cdot\sum_{X_n \in \{0, 1\}} P(X_1, X_2,..., X_n) = C,
$$
for some scalar $C$.

The proof is build in an interactive way, where the Prover and Verifier do a series of $n$ rounds. Each round is consists of a challange (created by the Verifier), computation (made by the Prover) and verifing (done by the Verifier). Using a Fiat-Shamir (FS) scheme, the proof become non-interactive. This allow the Prover to generate the entire proof and send it to Verifier which then verify the whole proof.

### Sumcheck with Combine Function
The Sumcheck protocol can be generalized to an arbitrary function of several multilinear polynomials. In this case, assuming there are $m$ polynomials of $n$ variables, and let $f$ be some arbitrary function that take $m$ polynomials and return a polynomial of higher (or equal) degree. Now the Prover tries to prove that:
$$
\sum_{X_1 \in \{0, 1\}}\sum_{X_2 \in \{0, 1\}}\cdot\cdot\cdot\sum_{X_n \in \{0, 1\}} f\left(P_1(X_1, ..., X_n), P_2(X_1, ..., X_n), ..., P_m(X_1, ..., X_n)\right) = C,
$$
for some scalar $C$.

## ICICLE's Sumecheck
ICICLE is implementing a non-onteractive Sumcheck protocol with support of a combine function. Sumcheck is supported by both CPU and CUDA backends of ICICLE. The polynomials are passed to the protocol as MLEs (evaluation representation).

### Implementation Limitations

There are some limitations / assumptions to the Sumcheck implementation.

- The maximum size of the polynomials (number of evaluations = $2^n$) depends on the number of polynomials and memory size of the device (e.g. CPU/GPU) used. For 4 polynomials one should expects that a GPU equiped with 24GB of memory can run Sumcheck with polynomials of size up to $2^{29}$.
- The polynomial size must be of size which is a power of 2.


#### CUDA Specific Limitations

There are some limitations spesific to the CUDA implementation of the Sumcheck protocol. Here is a list of them:

- The maximum number of polynomials is 4.
- The highest valid degree of the combined polynomial should be 4.
- The combine function can't be too complex. As a rule of thumb - the number of input polynomials + the number of constants used + the number of operations (e.g. addition, substruction and multlipication) should be less then 20.

## C++ API
A sumcheck is created by the following function:
```cpp
Sumcheck<scalar_t> create_sumcheck()
```

There are two configuration structs related to the Sumcheck protocol.

The `SumcheckConfig` struct is a configuration object used to specify parameters for Sumcheck. It contains the following fields:
- **`stream: icicleStreamHandle`**: Specifies the CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
- **`use_extension_field: bool`**: If true extension field is used for the Fiat-Shamir results. ***Currently not supported (should always be false)***
- **`batch: int`**: Number of input chuncks to hash in batch.
- **`are_inputs_on_device: bool`**: If true expect the input polynomials to reside on the device (e.g. GPU), if false expect them to reside on the host (e.g. CPU).
- **`is_async: bool`**: If true runs the hash asynchronously.
- **`ext: ConfigExtension*`**: Backend-specific extensions.

The default values are:
```cpp
  struct SumcheckConfig {
    icicleStreamHandle stream = nullptr;
    bool use_extension_field = false;
    uint64_t batch = 1;
    bool are_inputs_on_device =false;
    bool is_async = false;
    ConfigExtension* ext = nullptr;
  };
```

The `SumcheckTranscriptConfig<F>` class is a configuration object used to specify parameters to the Fiat-Shamir scheme used by the Sumcheck. It contained the following fields:
- **`hasher: Hash`**: Hash function used for randomness generation by Fiat-Shamir.
- **`domain_label: char*`**: Label for the domain separator in the transcript.
- **`poly_label: char*`**: Label for round polynomials in the transcript.
- **`challenge_label: char*`**: Label for round challenges in the transcript.
- **`seed: F`**: Seed for initializing the RNG.
- **`little_endian: bool`**: Encoding endianness

There are three constructors for `SumcheckTranscriptConfig<F>`, each one with its own arguments and default value.

The default constructor:
```cpp
    SumcheckTranscriptConfig()
        : m_little_endian(true), m_seed_rng(F::from(0)), m_hasher(std::move(create_keccak_256_hash()))
```

A constructor with byte vector for labels:
```cpp
    SumcheckTranscriptConfig(
      Hash hasher,
      std::vector<std::byte>&& domain_label,
      std::vector<std::byte>&& poly_label,
      std::vector<std::byte>&& challenge_label,
      F seed,
      bool little_endian = true)
        : m_hasher(std::move(hasher)), m_domain_separator_label(domain_label), m_round_poly_label(poly_label),
          m_round_challenge_label(challenge_label), m_little_endian(little_endian), m_seed_rng(seed)
```
A constructor with `const char*` arguments for labels
```cpp
    SumcheckTranscriptConfig(
      Hash hasher,
      const char* domain_label,
      const char* poly_label,
      const char* challenge_label,
      F seed,
      bool little_endian = true)
        : m_hasher(std::move(hasher)), m_domain_separator_label(cstr_to_bytes(domain_label)),
          m_round_poly_label(cstr_to_bytes(poly_label)), m_round_challenge_label(cstr_to_bytes(challenge_label)),
          m_little_endian(little_endian), m_seed_rng(seed)
```

### Generating Sumcheck Proofs
To generate a Proof, first and empty proof needed to be created. Sumceck proof is represented by the class `SumcheckProof<S>`:

```cpp
template <typename S>
class SumcheckProof
```

It has only the default constructor ` SumcheckProof()` which takes no arguments.

Then, the proof can be generate by the `get_proof` method of the `Sumcheck<F>` object:
```cpp
eIcicleError get_proof(
  const std::vector<F*>& mle_polynomials,
  const uint64_t mle_polynomial_size,
  const F& claimed_sum,
  const ReturningValueProgram<F>& combine_function,
  const SumcheckTranscriptConfig<F>&& transcript_config,
  const SumcheckConfig& sumcheck_config,
  SumcheckProof<F>& sumcheck_proof /*out*/) const
```

The arguments for this method are:
- **`mle_polynomials: std::vector<F*>&`**: A vector of pointer. Each pointer points to the evaluations of one of the input polynomials.
- **`mle_polynomial_size: uint64_t`**: The length of the polynomials (number of evaluations).
- **`claimed_sum: F&`**: The sum the claimed by the Prover to be the sum of the combine function over the evaluations of the polynomials.
- **`combine_function: ReturningValueProgram<F>&`**: The combine function. Uses ICICLE's [program](program.md) API.
- **`transcript_config: SumcheckTranscriptConfig<F>&&`**: The `SumcheckTranscriptConfig` object for the Fiat-Shamir scheme.
- **`sumcheck_config: SumcheckConfig&`**: The `SumcheckConfig` object for the Sumcheck configuration.
- **`sumcheck_proof: SumcheckProof<F>&`**: A `SumcheckProof` object which is the output of the `get_proof` method.

#### Example: Generating a Proof

Generating a Sumcheck proof:

```cpp
auto prover_sumcheck = create_sumcheck<scalar_t>();
SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

ReturningValueProgram<scalar_t> combine_func(EQ_X_AB_MINUS_C);
SumcheckConfig sumcheck_config;
SumcheckProof<scalar_t> sumcheck_proof;
ICICLE_CHECK(prover_sumcheck.get_proof(
  mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
  sumcheck_proof));
```

### Verifying Sumcheck Proofs

To verify the proof, the Verifier should use the method `verify` of the `Sumcheck<F>` object:

```cpp
eIcicleError verify(
  const SumcheckProof<F>& sumcheck_proof,
  const F& claimed_sum,
  const SumcheckTranscriptConfig<F>&& transcript_config,
  bool& valid /*out*/)
```

The arguments for this method are:
- **`sumcheck_proof: SumcheckProof<F>&`**: The proof the verifier wants to verify.
- **`claimed_sum: F&`**: The sum, claimed by the Prover, the Verifier wants to check.
- **`transcript_config: SumcheckTranscriptConfig<F>&&`**: The `SumcheckTranscriptConfig` object for the Fiat-Shamir scheme.
- **`valid: bool`**: The output of the method. True if the proof was verified correctly, false otherwise.

> **_NOTE:_**  The `SumcheckTranscriptConfig` used for generating the proof should be **identical** to the one used to verify it.

#### Example: Verifying a Proof

Verifying a Shumcheck proof:

```cpp
auto verifier_sumcheck = create_sumcheck<scalar_t>();
bool verification_pass = false;
ICICLE_CHECK(
  verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass));
```

After `verifier_sumcheck.verify` the variable `verification_pass` is `true` if the proof is valid and `false` otherwise.