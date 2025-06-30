# Lattice-Based SNARKs — Rust API Overview

## Overview

ICICLE provides a modular, high-performance Rust API for lattice-based SNARK constructions. Implemented across the `icicle-core` and `icicle-labrador` crates, the API supports efficient operations over integer and polynomial rings, with CPU and CUDA backends.

The design is generic over ring constructions, enabling flexible use of different `Zq` and `Rq` instantiations for cryptographic protocols like **Labrador**.

## Key Capabilities

- **Core Types**
  - `Zq`: Integer rings modulo \( q \)
  - `Rq` / `Tq`: Polynomial rings `Zq[X]/(Xⁿ + 1)`

- **Algebraic Operations**
  - Number-Theoretic Transforms (NTT)
  - Matrix multiplication and transpose
  - Balanced base decomposition
  - JL projection

- **Vector and Ring Operations**
  - Elementwise addition, subtraction, multiplication, sum-reduce

- **Cryptographic Primitives**
  - Norm computation (ℓ₂ and operator norm)
  - Random vector sampling
  - Challenge sampling with rejection based on operator norm

## Design Highlights

- Generic over ring types (`Zq`, `Rq`, etc.)
- Unified support for various ICICLE backends
- Optimized for batching and aggregation
- Extensible for lattice-based SNARKs and post-quantum cryptography

## Core Types

### Integer Ring: Zq

The integer ring `Zq` represents integers modulo `q`, where `q` is typically a product of small prime fields for efficiency.

// TODO show q


// TODO: verify examples compile and work correctly before merging!
// TODO: replace labrador with the final name that

```rust
use icicle_labrador::ring::{ScalarRing as Zq};
use icicle_core::traits::{FieldImpl, GenerateRandom};

// Generate random Zq elements
let zq_random: Vec<Zq> = Zq::generate_random(size);
// Generate zeros Zq elements
let zq_zeros: Vec<Zq> = vec![Zq::zero(); size];
// Generate elements from arbitrary bytes
let element_size = std::mem::size_of::<Zq>();
let zq_from_bytes = user_bytes
    .chunks(element_size)
    .map(Zq::from_bytes_le)
    .collect()
```

### Polynomial Ring: Rq

The polynomial ring `Rq = Zq[X]/(X^d + 1)` represents polynomials of degree less than `d` with coefficients in `Zq`.

```rust
use icicle_labrador::polynomial_ring::PolyRing;
use icicle_core::traits::GenerateRandom;

// Polynomial ring Rq = Zq[X]/(X^d + 1) where d = 64
type PolyRing = icicle_labrador::polynomial_ring::PolyRing;
type Rq = PolyRing;  // Alias for coefficient domain
type Tq = PolyRing;  // Alias for NTT domain

// Generate random polynomials
let size = 8;
let rq_random: Vec<PolyRing> = PolyRing::generate_random(size);
let rq_zeros: Vec<PolyRing> = vec![PolyRing::zero(); size];

// Convert Zq chunks to Rq polynomials
let rq_from_slice: Vec<PolyRing> = zq_random
    .chunks(PolyRing::DEGREE)
    .map(PolyRing::from_slice)
    .collect();
```

## Negacyclic NTT

### Forward and Inverse NTT (Rq to/from Tq)

The negacyclic Number-Theoretic Transform (NTT) converts polynomials in `Rq = Zq[X]/(Xⁿ + 1)` to the evaluations domain `Tq`, enabling efficient polynomial multiplication via element-wise operations.

### Main Imports

To use the NTT API, import the following symbols:

```rust
use icicle_core::negacyclic_ntt::{
    NegacyclicNttConfig, // Configuration for backend/device/async
    NegacyclicNtt,       // Trait implemented by polynomial types
    ntt,                 // Out-of-place NTT wrapper
    ntt_inplace,         // In-place NTT wrapper
    NTTDir,              // Transform direction (Forward or Inverse)
};
```

### API

```rust
/// Performs a negacyclic Number-Theoretic Transform (NTT) over a polynomial ring.
///
/// - `input`: Input slice containing polynomials
/// - `output`: Output slice to store the transformed result
/// - `dir`: Transform direction (`Forward` or `Inverse`)
/// - `cfg`: Execution configuration (device flags, stream, async mode)
pub fn ntt<P: PolynomialRing + NegacyclicNtt<P>>(
    input: &(impl HostOrDeviceSlice<P> + ?Sized),
    dir: NTTDir,
    cfg: &NegacyclicNttConfig,
    output: &mut (impl HostOrDeviceSlice<P> + ?Sized),
) -> Result<(), eIcicleError> {
    P::ntt(input, dir, cfg, output)
}

/// Performs an in-place negacyclic NTT over a polynomial ring.
///
/// Performs an in-place negacyclic Number-Theoretic Transform (NTT) over a polynomial ring
///
/// - `inout`: Buffer to transform in-place
/// - `dir`: Transform direction (`Forward` or `Inverse`)
/// - `cfg`: Execution configuration
pub fn ntt_inplace<P: PolynomialRing + NegacyclicNtt<P>>(
    inout: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    dir: NTTDir,
    cfg: &NegacyclicNttConfig,
) -> Result<(), eIcicleError> {
    P::ntt_inplace(inout, dir, cfg)
}
```

### Example:

```rust
use icicle_labrador::PolyRing;
// Generate random input on the host
let input = PolyRing::generate_random(size); 

// Allocate and transfer to device memory
let mut device_input = DeviceVec::<PolyRing>::device_malloc(size)?;
device_input.copy_from_host(HostSlice::from_slice(&input))?;

// Compute in-place (or out of place)
let config = NegacyclicNttConfig::default();    
negacyclic_ntt::ntt_inplace(&mut device_input, NTTDir::kForward, &config);    
```

## Matrix Operations

### Matrix Multiplication and Transpose over Ring Elements

ICICLE provides generic APIs for performing matrix operations over ring elements such as `Zq` and `PolyRing`.

Supported operations include:

- Dense matrix multiplication (row-major layout)
- Matrix transposition (row-major input)

These are useful for vector dot-products, Ajtai-style commitments, and other algebraic primitives in lattice-based SNARKs.

---

### Main Imports

```rust
use icicle_core::matrix_ops::{
    matmul,                 // Matrix multiplication
    matrix_transpose,       // Matrix transpose
    VecOpsConfig,           // Backend and execution configuration
    MatrixOps,              // Trait for matmul    
};
```


### Matrix multiplication API

```rust
/// Computes C = A × B for two row-major matrices.
///
/// - `a`: Matrix A, shape (a_rows × a_cols)
/// - `b`: Matrix B, shape (b_rows × b_cols)
/// - `cfg`: Execution configuration
/// - `result`: Output buffer, shape (a_rows × b_cols)
pub fn matmul<T>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    a_rows: u32,
    a_cols: u32,
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    b_rows: u32,
    b_cols: u32,
    cfg: &VecOpsConfig,
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>;
```

### Matrix transpose API

```rust
/// Transposes a row-major matrix: result(i, j) = input(j, i)
///
/// - `input`: Source matrix of shape (rows × cols)
/// - `result`: Output buffer of shape (cols × rows)
/// - `cfg`: Execution configuration
pub fn matrix_transpose<T>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    nof_rows: u32,
    nof_cols: u32,
    cfg: &VecOpsConfig,
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>;
```

### Example

```rust
use icicle_labrador::PolyRing;
use icicle_core::matrix_ops::{matmul, matrix_transpose, VecOpsConfig};
use icicle_runtime::memory::{DeviceVec, HostSlice};

let n = 8;
let m = 64;
let cfg = VecOpsConfig::default();

// Generate a random matrix A ∈ [n × m] on the host (row-major layout)
let host_a = PolyRing::generate_random((n * m) as usize);

// Allocate device buffer for Aᵗ ∈ [m × n]
let mut device_a_transposed = DeviceVec::<PolyRing>::device_malloc((n * m) as usize)
    .expect("Failed to allocate transpose output");

// Transpose Aᵗ = transpose(A) (from host memory to device memory)
matrix_transpose(HostSlice::from_slice(&host_a), n, m, &cfg, &mut device_a_transposed)
    .expect("Transpose failed");

// Allocate output buffer for (Aᵗ)A ∈ [m × m]
let mut device_a_transposed_a = DeviceVec::<PolyRing>::device_malloc((m * m) as usize)
    .expect("Failed to allocate output matrix");

// Compute (Aᵗ)A
// Note that one matrix is on host memory and the other on device memory
matmul(&device_a_transposed, m, n, HostSlice::from_slice(&host_a), n, m, &cfg, &mut device_a_transposed_a)
    .expect("Matmul failed");
```

## Balanced Base Decomposition

### Decompose and Recompose Ring Elements

Balanced base decomposition expresses each ring element (e.g. `Zq`, `Rq`) as a sequence of digits in a given base `b`, where each digit lies in the interval `(-b/2, b/2]`.

---

### Output Layout

For an input slice of `n` elements and a digit count `d = count_digits(base)`:

- The output vector has length `n × d`.
- The layout is **digit-major** (not element-major):
  - The first `n` entries are the **first digit** of all elements.
  - The next `n` entries are the **second digit** of all elements.
  - And so on, until all `d` digits are emitted.
- Conceptually, this forms a matrix of shape `[d × n]`, where each row corresponds to a digit index and each column to an element.
- If you allocate fewer than `d` digit rows (i.e., a shorter output buffer), decomposition will truncate early.
  - **Warning**: recomposition will only reconstruct the original values correctly if all omitted most significant digits were zero.
---

### Main Imports

```rust
use icicle_core::balanced_decomposition::{
    decompose,            // Decomposition function
    recompose,            // Recomposition function
    count_digits,         // Compute number of digits needed
    BalancedDecomposition // Trait for custom rings
};
```

### API

```rust
/// Returns the number of digits required to represent a ring element in balanced base-`b` form.
///
/// Each digit lies in the interval (-b/2, b/2], and the number of digits depends on the modulus.
fn count_digits<T: BalancedDecomposition<T>>(base: u32) -> u32

/// Decomposes a slice of elements into balanced base-`b` digits (digit-major layout).
///
/// - `input.len()` = number of elements to decompose
/// - `output.len()` must be `input.len() × num_digits`, where `num_digits ∈ [1, count_digits(base)]`
///
/// Digits are written in order of increasing significance, grouped by digit index.
fn decompose<T: BalancedDecomposition<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>

/// Recomposes original elements from digit-major base-`b` decomposition.
///
/// - `input.len()` must be `output.len() × num_digits`, where `num_digits ∈ [1, count_digits(base)]`
///
/// Recomposition is exact only if all omitted higher-order digits were zero.
fn recompose<T: BalancedDecomposition<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
```

### Examples

```rust
let base = 4; // Typically set to q^(1/t) for small t (e.g., t = 2, 4, 6)
let size = 1024;
// Compute number of digits per element for the given base
let digits = balanced_decomposition::count_digits::<P>(base);
let output_len = size * digits as usize;

// Allocate device memory for digit-major output
let mut decomposed = DeviceVec::<P>::device_malloc(output_len)
    .expect("Failed to allocate device memory");

// Generate input vector
let input = P::generate_random(size);

// Perform balanced base decomposition
let cfg = VecOpsConfig::default();
balanced_decomposition::decompose::<P>(
    HostSlice::from_slice(&input),
    &mut decomposed,
    base,
    &cfg,
).expect("Decomposition failed");
```


## Norm Checking

### L2 and L∞ Norm Verification

```rust
use icicle_core::norm::{Norm, NormType};

// Check if vector satisfies norm bound
norm::check_norm_bound(
    &device_input,
    NormType::L2,           // Type of norm to check
    upper_bound,            // Norm bound to check against
    &config,                // Configuration
    HostSlice::from_mut_slice(&mut output), // Output: true if bound satisfied
)?;
```

### Example: Norm Checking

```rust
use icicle_core::{
    norm::{Norm, NormType},
    traits::GenerateRandom,
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

fn norm_checking_example<T>(size: usize)
where
    T: FieldImpl,
    <T as FieldImpl>::Config: Norm<T> + GenerateRandom<T>,
{
    let max_val: u32 = 1 << 30;
    let mut l2_squared: u128 = 0;
    let mut l_infinity_norm: u32 = 0;

    // Generate random input and compute norms
    let input: Vec<T> = (0..size)
        .map(|_| {
            let rand_val: u32 = rand::thread_rng().gen_range(0..=max_val);
            l2_squared += (rand_val as u128) * (rand_val as u128);
            l_infinity_norm = l_infinity_norm.max(rand_val);
            T::from_u32(rand_val)
        })
        .collect();

    let l2_norm: u64 = l2_squared.isqrt() as u64;

    // Upload to device
    let mut device_input = DeviceVec::<T>::device_malloc(size)?;
    device_input.copy_from_host(HostSlice::from_slice(&input))?;

    let config = VecOpsConfig::default();

    // ℓ₂ norm check — upper bound (should pass)
    let mut output = vec![false; 1];
    let upper_bound = l2_norm + 1;
    norm::check_norm_bound(
        &device_input,
        NormType::L2,
        upper_bound,
        &config,
        HostSlice::from_mut_slice(&mut output),
    )?;
    assert!(output[0], "ℓ₂ norm check failed unexpectedly");

    // ℓ₂ norm check — tight bound (should fail)
    let lower_bound = l2_norm;
    norm::check_norm_bound(
        &device_input,
        NormType::L2,
        lower_bound,
        &config,
        HostSlice::from_mut_slice(&mut output),
    )?;
    assert!(!output[0], "ℓ₂ norm check unexpectedly passed");

    // ℓ∞ norm check for batch vectors
    let batch = 4;
    let mut output = vec![false; batch];
    norm::check_norm_bound(
        &device_input,
        NormType::LInfinity,
        l_infinity_norm as u64 + 1,
        &config,
        HostSlice::from_mut_slice(&mut output),
    )?;
    assert!(output.iter().all(|&x| x), "ℓ∞ norm check failed");
}
```

## Johnson-Lindenstrauss Projection

### JL Projection and Matrix Row Generation

```rust
use icicle_core::jl_projection::{JLProjection, JLProjectionPolyRing};

// Perform JL projection
jl_projection::jl_projection(
    &zq_device_slice,       // Input vector (flattened Zq)
    &seed,                  // Random seed
    &config,                // Configuration
    &mut device_output      // Output vector
)?;

// Get JL matrix rows as polynomial ring elements
jl_projection::get_jl_matrix_rows_as_polyring(
    &seed,                  // Random seed
    row_size,               // Number of input polynomials per row
    0,                      // Starting row index
    num_rows,               // Number of rows to generate
    true,                   // conjugated = true
    &config,                // Configuration
    &mut jl_rows            // Output matrix rows
)?;
```

### Example: JL Projection

```rust
use icicle_core::{
    jl_projection::{JLProjection, JLProjectionPolyRing},
    polynomial_ring::{flatten_polyring_slice, flatten_polyring_slice_mut},
    traits::GenerateRandom,
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

fn jl_projection_example<P>(size: usize, projection_dim: usize)
where
    P: PolynomialRing + GenerateRandom<P>,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: JLProjection<P::Base>,
    P: JLProjectionPolyRing<P>,
{
    let config = VecOpsConfig::default();
    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    // Generate input and copy to device
    let host_input: Vec<P> = P::generate_random(size);
    let mut device_input = DeviceVec::<P>::device_malloc(size)?;
    device_input.copy_from_host(HostSlice::from_slice(&host_input))?;

    // JL projection on flattened device memory
    let zq_device_slice = flatten_polyring_slice(&device_input);
    let mut device_output = DeviceVec::<P::Base>::device_malloc(projection_dim)?;

    let t_start = std::time::Instant::now();
    jl_projection::jl_projection(&zq_device_slice, &seed, &config, &mut device_output)?;
    let t_elapsed = t_start.elapsed();

    // Retrieve conjugated JL matrix rows as PolyRing polynomials
    let row_size = size;
    let num_rows = 1;
    let mut jl_rows = DeviceVec::<P>::device_malloc(num_rows * row_size)?;

    let t_start = std::time::Instant::now();
    jl_projection::get_jl_matrix_rows_as_polyring(
        &seed,
        row_size,
        0,
        num_rows,
        true, // conjugated = true
        &config,
        &mut jl_rows,
    )?;
    let t_elapsed = t_start.elapsed();
}
```

## Random Sampling

### Seeded Random Generation

```rust
use icicle_core::random_sampling::RandomSampling;

// Generate pseudorandom elements
random_sampling::random_sampling(
    fast_mode,              // Use fast sampling mode
    &seed,                  // Random seed
    &config,                // Configuration
    &mut output_zq          // Output array
)?;
```

### Example: Random Sampling

```rust
use icicle_core::{
    random_sampling::RandomSampling,
    polynomial_ring::flatten_polyring_slice_mut,
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

fn random_sampling_example<P>(size: usize)
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: RandomSampling<P::Base>,
{
    let mut seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut seed);

    let fast_mode = true;
    let config = VecOpsConfig::default();

    // Sample Zq elements
    let mut output_zq = DeviceVec::<P::Base>::device_malloc(size)?;
    let start = std::time::Instant::now();
    random_sampling::random_sampling(fast_mode, &seed, &config, &mut output_zq)?;
    let duration = start.elapsed();

    // Sample Rq polynomials by reinterpreting as Zq elements
    let mut output_rq = DeviceVec::<P>::device_malloc(size)?;
    let mut output_rq_coeffs = flatten_polyring_slice_mut(&mut output_rq);

    let start = std::time::Instant::now();
    random_sampling::random_sampling(fast_mode, &seed, &config, &mut output_rq_coeffs)?;
    let duration = start.elapsed();
}
```

## Vector Operations for Polynomial Rings

### Polynomial Vector Operations

```rust
use icicle_core::vec_ops::poly_vecops;

// Element-wise addition of polynomial vectors
poly_vecops::polyvec_add(&polyvec, &polyvec_b, &mut result, &config)?;

// Element-wise subtraction of polynomial vectors
poly_vecops::polyvec_sub(&polyvec, &polyvec_b, &mut result, &config)?;

// Element-wise multiplication of polynomial vectors
poly_vecops::polyvec_mul(&polyvec, &polyvec_b, &mut result, &config)?;

// Scalar multiplication of polynomial vector
poly_vecops::polyvec_mul_by_scalar(&polyvec, &scalarvec, &mut result, &config)?;

// Sum reduction of polynomial vector
poly_vecops::polyvec_sum_reduce(&mul_result, &mut reduced, &config)?;
```

### Example: Vector Operations

```rust
use icicle_core::{
    vec_ops::{poly_vecops, VecOpsConfig},
    random_sampling::RandomSampling,
    polynomial_ring::flatten_polyring_slice_mut,
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

fn polynomial_vecops_example<P>(size: usize)
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base> + RandomSampling<P::Base>,
{
    let config = VecOpsConfig::default();
    let fast_mode = true;

    // Generate a random seed
    let mut seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut seed);

    // Allocate and sample a vector of polynomials
    let mut polyvec = DeviceVec::<P>::device_malloc(size)?;
    {
        let mut polyvec_flat = flatten_polyring_slice_mut(&mut polyvec);
        random_sampling::random_sampling(fast_mode, &seed, &config, &mut polyvec_flat)?;
    }

    // Allocate and sample a vector of scalars
    let mut scalarvec = DeviceVec::<P::Base>::device_malloc(size)?;
    random_sampling::random_sampling(fast_mode, &seed, &config, &mut scalarvec)?;

    // Allocate result buffer for the pointwise multiplication
    let mut mul_result = DeviceVec::<P>::device_malloc(size)?;

    // Perform polyvec_mul_by_scalar
    let start = std::time::Instant::now();
    poly_vecops::polyvec_mul_by_scalar(&polyvec, &scalarvec, &mut mul_result, &config)?;
    println!("polyvec_mul_by_scalar completed in {:?}", start.elapsed());

    // Allocate output for sum-reduction into a single polynomial
    let mut reduced = DeviceVec::<P>::device_malloc(1)?;

    // Reduce with polyvec_sum_reduce
    let start = std::time::Instant::now();
    poly_vecops::polyvec_sum_reduce(&mul_result, &mut reduced, &config)?;
    println!("polyvec_sum_reduce completed in {:?}", start.elapsed());
}
```

## Parameters and Configuration

### Labrador Configuration

The Labrador protocol uses specific parameters optimized for lattice-based SNARKs:

```rust
use icicle_labrador::{
    polynomial_ring::PolyRing,
    ring::{ScalarCfg as ZqCfg, ScalarRing as Zq},
};

// Integer ring configuration
// Zq represents the integer ring Z/qZ where q = P_bb * P_kb
type Zq = ScalarRing<ScalarCfg>;

// Polynomial ring configuration
// Rq = Zq[X]/(X^d + 1) where d = 64
type PolyRing = icicle_labrador::polynomial_ring::PolyRing;
type Rq = PolyRing;
type Tq = PolyRing;
```

### Performance Considerations

- **Memory Layout**: Polynomial ring operations use digit-major layout for balanced decomposition
- **Batch Processing**: Vector operations support batch processing for improved throughput
- **Device Memory**: Operations can be performed on CPU or GPU with automatic memory management
- **Async Operations**: Support for asynchronous execution using CUDA streams

### Error Handling

All functions return `Result<T, E>` where `E` implements `std::error::Error`:

```rust
// Example error handling
match negacyclic_ntt::ntt_inplace(&mut device_input, NTTDir::kForward, &config) {
    Ok(()) => println!("NTT completed successfully"),
    Err(e) => eprintln!("NTT failed: {}", e),
}
```

## Complete Example

Here's a complete example demonstrating key lattice SNARK operations:

```rust
use icicle_core::{
    balanced_decomposition, jl_projection, matrix_ops, negacyclic_ntt,
    ntt::NTTDir,
    polynomial_ring::{flatten_polyring_slice, flatten_polyring_slice_mut, PolynomialRing},
    random_sampling,
    traits::{Arithmetic, FieldImpl, GenerateRandom},
    vec_ops,
    vec_ops::{poly_vecops, VecOpsConfig},
};
use icicle_labrador::{
    polynomial_ring::PolyRing,
    ring::{ScalarCfg as ZqCfg, ScalarRing as Zq},
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

fn lattice_snark_example() -> Result<(), Box<dyn std::error::Error>> {
    let size = 1 << 10; // Adjustable test size

    // 1. Integer ring: Zq
    let zq_random: Vec<Zq> = ZqCfg::generate_random(size);
    println!("[Integer ring Zq] Generated {} random Zq elements", zq_random.len());

    // 2. Polynomial ring: PolyRing = Zq[X]/(X^n + 1)
    let rq_random: Vec<PolyRing> = PolyRing::generate_random(size);
    let rq_from_slice: Vec<PolyRing> = zq_random
        .chunks(PolyRing::DEGREE)
        .map(PolyRing::from_slice)
        .collect();
    println!("[Polynomial Ring Rq] Converted {} Zq chunks into Rq polynomials", rq_from_slice.len());

    // 3. Negacyclic NTT for polynomial rings
    negacyclic_ntt_example::<PolyRing>(size)?;

    // 4. Polynomial Ring Matrix Multiplication
    matmul_example::<PolyRing>(size as u32 >> 3, size as u32, size as u32 >> 2)?;

    // 5. Balanced base decomposition for polynomial rings
    balanced_decomposition_example::<PolyRing>(size)?;

    // 6. Norm Checking for Integer Ring (Zq)
    norm_checking_example::<Zq>(size)?;

    // 7. Johnson–Lindenstrauss Projection for Zq
    jl_projection_example::<PolyRing>(size, 256)?;

    // 8. Vector APIs for Polynomial Rings
    polynomial_vecops_example::<PolyRing>(size)?;

    // 9. Matrix Transpose for Polynomial Rings
    transpose_example::<PolyRing>(size as u32, size as u32 >> 2)?;

    // 10. Random Sampling for Zq and PolyRing
    random_sampling_example::<PolyRing>(size)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================== Lattice SNARK Example ====================");
    
    // Set up device (CPU or CUDA)
    let device_type = std::env::args().nth(1).unwrap_or_else(|| "CPU".to_string());
    if device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default()?;
    }
    let device = icicle_runtime::Device::new(&device_type, 0);
    icicle_runtime::set_device(&device)?;
    
    lattice_snark_example()?;
    
    Ok(())
}
```

## Running the Example

To run the lattice SNARKs example:

```bash
# Run on CPU
cargo run --release

# Run on CUDA
cargo run --release --features cuda -- --device-type CUDA
```

This documentation covers the complete Rust API for lattice-based SNARKs in ICICLE, providing all the necessary types, functions, and examples for implementing protocols like Labrador. 