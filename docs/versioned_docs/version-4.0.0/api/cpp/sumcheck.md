# Sumcheck API Documentation

## Overview
The Sumcheck protocol allows a Prover to prove to a Verifier that the sum of a multilinear polynomial, over the Boolean hypercube, equals a specific scalar value.

For a polynomial $P$ with $n$ variables, the Prover aims to prove that:
$$
\sum_{X_1 \in \{0, 1\}}\sum_{X_2 \in \{0, 1\}}\cdot\cdot\cdot\sum_{X_n \in \{0, 1\}} P(X_1, X_2,..., X_n) = C,
$$
where $C$ is some scalar.

The proof is constructed interactively, involving a series of $n$ rounds. Each round consists of a challenge (from the Verifier), a computation (from the Prover), and a verification step (by the Verifier). Using a Fiat-Shamir (FS) scheme, the proof becomes non-interactive, enabling the Prover to generate the entire proof and send it to the Verifier for validation.

### Sumcheck with Combine Function
The Sumcheck protocol can be generalized to handle multiple multilinear polynomials. In this case, assuming there are $m$ polynomials of $n$ variables, and $f$ is an arbitrary function that takes these $m$ polynomials and returns a polynomial of equal or higher degree, the Prover tries to prove that:
$$
\sum_{X_1 \in \{0, 1\}}\sum_{X_2 \in \{0, 1\}}\cdot\cdot\cdot\sum_{X_n \in \{0, 1\}} f\left(P_1(X_1, ..., X_n), P_2(X_1, ..., X_n), ..., P_m(X_1, ..., X_n)\right) = C,
$$
where $C$ is some scalar.

## ICICLE's Sumecheck Implementation
ICICLE implements a non-interactive Sumcheck protocol that supports a combine function. It is available on both the CPU and CUDA backends of ICICLE. The polynomials are passed to the protocol in MLE (evaluation representation) form.

### Implementation Limitations

There are some limitations and assumptions in the Sumcheck implementation:

- The maximum size of the polynomials (i.e., the number of evaluations $2^n$) depends on the number of polynomials and the memory size of the device (e.g., CPU/GPU) being used. For example, with 4 polynomials, a GPU with 24GB of memory should be able to handle Sumcheck for polynomials of size up to $2^29$.
- The polynomial size must a power of 2.
- The current implementation does not support generating the challenge ($\alpha$) in an extension field. This functionality is necessary for ensuring security when working with small fields.

## C++ API
A Sumcheck object can be created using the following function:
```cpp
Sumcheck<scalar_t> create_sumcheck()
```

There are two key configuration structs related to the Sumcheck protocol.

### SumcheckConfig
The `SumcheckConfig` struct is used to specify parameters for the Sumcheck protocol. It contains the following fields:
- **`stream: icicleStreamHandle`**: The CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
- **`use_extension_field: bool`**: If true, an extension field is used for Fiat-Shamir results. Currently unsupported (should always be false).
- **`are_inputs_on_device: bool`**: If true, the input polynomials are expected to reside on the device (e.g., GPU); otherwise, they are expected to reside on the host (e.g., CPU).
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

### SumcheckTranscriptConfig
The `SumcheckTranscriptConfig<F>` class is used to specify parameters for the Fiat-Shamir scheme used by the Sumcheck protocol. It contains the following fields:
- **`hasher: Hash`**: The hash function used to generate randomness for Fiat-Shamir.
- **`domain_label: char*`**: The label for the domain separator in the transcript.
- **`poly_label: char*`**: The label for round polynomials in the transcript.
- **`challenge_label: char*`**: The label for round challenges in the transcript.
- **`seed: F`**: The seed for initializing the RNG.
- **`little_endian: bool`**: The encoding endianness.

There are three constructors for `SumcheckTranscriptConfig<F>`, each with its own default values:

* **Default constructor**:
```cpp
    SumcheckTranscriptConfig()
        : m_little_endian(true), m_seed_rng(F::from(0)), m_hasher(std::move(create_keccak_256_hash()))
```

* **Constructor with byte vector for labels**:
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

* **Constructor with `const char*` arguments for labels**:
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
To generate a proof, first, an empty proof needs to be created. The Sumcheck proof is represented by the `SumcheckProof<S>` class:

```cpp
template <typename S>
class SumcheckProof
```

The class has a default constructor `SumcheckProof()` that takes no arguments.

The proof can be generated using the get_proof method from the `Sumcheck<F>` object:
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
- **`mle_polynomials: std::vector<F*>&`**: A vector of pointers, each pointing to the evaluations of one of the input polynomials.
- **`mle_polynomial_size: uint64_t`**: The length of the polynomials (number of evaluations). This should be a power of 2.
- **`claimed_sum: F&`**: The sum the Prover claims to be the sum of the combine function over the evaluations of the polynomials.
- **`combine_function: ReturningValueProgram<F>&`**: The combine function, using ICICLE's [program](program.md) API.
- **`transcript_config: SumcheckTranscriptConfig<F>&&`**: The configuration for the Fiat-Shamir scheme.
- **`sumcheck_config: SumcheckConfig&`**: The configuration for the Sumcheck protocol.
- **`sumcheck_proof: SumcheckProof<F>&`**: The output `SumcheckProof` object containing the generated proof.

#### Example: Generating a Proof

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

To verify the proof, the Verifier should use the verify method of the `Sumcheck<F>` object:

```cpp
eIcicleError verify(
  const SumcheckProof<F>& sumcheck_proof,
  const F& claimed_sum,
  const SumcheckTranscriptConfig<F>&& transcript_config,
  bool& valid /*out*/)
```

The arguments for this method are:
- **`sumcheck_proof: SumcheckProof<F>&`**: The proof that the Verifier wants to verify.
- **`claimed_sum: F&`**: The sum that the Verifier wants to check, claimed by the Prover.
- **`transcript_config: SumcheckTranscriptConfig<F>&&`**: The configuration for the Fiat-Shamir scheme.
- **`valid: bool`**: The output of the method. `true` if the proof is valid, `false` otherwise.

> **_NOTE:_**  The `SumcheckTranscriptConfig` used for generating the proof must be identical to the one used for verification.

#### Example: Verifying a Proof
```cpp
auto verifier_sumcheck = create_sumcheck<scalar_t>();
bool verification_pass = false;
ICICLE_CHECK(
  verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass));
```

After calling `verifier_sumcheck.verify`, the variable `verification_pass` will be `true` if the proof is valid, and `false` if not.