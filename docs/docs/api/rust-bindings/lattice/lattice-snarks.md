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

### Matrix Multiplication

```rust
use icicle_core::matrix_ops::MatrixOps;

// Matrix multiplication for polynomial rings
matrix_ops::matmul::<P>(
    &device_a,      // Input matrix A [n × m]
    n,              // Number of rows in A
    m,              // Number of columns in A
    &device_b,      // Input matrix B [m × k]
    m,              // Number of rows in B
    k,              // Number of columns in B
    &config,        // Configuration
    &mut device_c   // Output matrix C [n × k]
)?;
```

### Matrix Transpose

```rust
use icicle_core::matrix_ops::MatrixOps;

// Matrix transpose for polynomial rings
matrix_ops::transpose::<P>(
    &device_input,  // Input matrix [rows × cols]
    rows,           // Number of rows
    cols,           // Number of columns
    &config,        // Configuration
    &mut device_output // Output matrix [cols × rows]
)?;
```

### Example: Matrix Operations

```rust
use icicle_core::{
    matrix_ops::MatrixOps,
    traits::GenerateRandom,
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

fn matmul_example<P>(n: u32, m: u32, k: u32)
where
    P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>,
{
    let config = VecOpsConfig::default();
    let a_len = (n * m) as usize;
    let b_len = (m * k) as usize;
    let c_len = (n * k) as usize;

    // Generate random host-side input matrices
    let host_a: Vec<P> = P::generate_random(a_len);
    let host_b: Vec<P> = P::generate_random(b_len);

    // Allocate device memory for inputs and output
    let mut device_a = DeviceVec::<P>::device_malloc(a_len)?;
    let mut device_b = DeviceVec::<P>::device_malloc(b_len)?;
    let mut device_c = DeviceVec::<P>::device_malloc(c_len)?;

    // Transfer inputs to device
    device_a.copy_from_host(HostSlice::from_slice(&host_a))?;
    device_b.copy_from_host(HostSlice::from_slice(&host_b))?;

    // Perform matrix multiplication on device: C = A × B
    let start = std::time::Instant::now();
    matrix_ops::matmul::<P>(&device_a, n, m, &device_b, m, k, &config, &mut device_c)?;
    let elapsed = start.elapsed();
    println!("[Matmul] Completed in {:?}", elapsed);
}
```

## Balanced Decomposition

### Decomposition and Recomposition

```rust
use icicle_core::balanced_decomposition::BalancedDecomposition;

// Compute number of digits needed for base-b decomposition
let digits_per_elem = balanced_decomposition::count_digits::<P>(base);

// Decompose elements into balanced base-b digits
balanced_decomposition::decompose::<P>(
    HostSlice::from_slice(&input),
    &mut decomposed[..],
    base,
    &config,
)?;

// Recompose elements from balanced base-b digits
balanced_decomposition::recompose::<P>(
    &decomposed[..],
    HostSlice::from_mut_slice(&mut recomposed),
    base,
    &config,
)?;
```

### Example: Balanced Decomposition

```rust
use icicle_core::{
    balanced_decomposition::BalancedDecomposition,
    traits::GenerateRandom,
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

fn balanced_decomposition_example<P>(size: usize)
where
    P: PolynomialRing + BalancedDecomposition<P> + GenerateRandom<P>,
    P::Base: FieldImpl + Arithmetic,
{
    let q = modulus::<P::Base>();
    let ts = [2, 4, 6];
    let bases: Vec<u32> = ts
        .iter()
        .map(|t| (q as f64).powf(1.0 / *t as f64).floor() as u32)
        .collect();

    // Generate input data
    let input = P::generate_random(size);
    let mut recomposed = vec![P::zero(); size];
    let config = VecOpsConfig::default();

    for (i, base) in bases.iter().enumerate() {
        let digits_per_elem = balanced_decomposition::count_digits::<P>(*base);
        let decomposed_len = size * digits_per_elem as usize;

        let mut decomposed = DeviceVec::<P>::device_malloc(decomposed_len)?;

        // Decompose
        let t0 = std::time::Instant::now();
        balanced_decomposition::decompose::<P>(
            HostSlice::from_slice(&input),
            &mut decomposed[..],
            *base,
            &config,
        )?;
        let decompose_time = t0.elapsed();

        // Recompose
        let t1 = std::time::Instant::now();
        balanced_decomposition::recompose::<P>(
            &decomposed[..],
            HostSlice::from_mut_slice(&mut recomposed),
            *base,
            &config,
        )?;
        let recompose_time = t1.elapsed();

        // Verification
        assert_eq!(input, recomposed);
    }
}
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