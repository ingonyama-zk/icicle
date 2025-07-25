
# Lattices — Rust API Overview

## Overview

ICICLE provides a modular, high-performance Rust API for lattice-based SNARK constructions. Implemented across the `icicle-core` and `icicle-babykoala` crates, the API supports efficient operations over integer and polynomial rings, with CPU and CUDA backends.

The design is generic over ring constructions, enabling flexible use of different `Zq` and `Rq` instantiations for cryptographic protocols like **labrador**.

## Key Capabilities

### Core Types

- **`Zq`** — Integer rings modulo \( q \)
- **`Rq` / `Tq`** — Polynomial rings `Zq[X]/(Xⁿ + 1)`
  - `Rq` refers to the coefficient (standard) representation.
  - `Tq` refers to the evaluation (NTT-transformed) representation.
  - In ICICLE, both share a single unified trait.

### Supported Operations

- **Negacyclic Number-Theoretic Transforms (NTT)**  
  For fast polynomial multiplication in `Tq`
- **Matrix Operations**  
  Matrix multiplication and transpose
- **Vector Operations**  
  Elementwise arithmetic, sum-reduction, scalar ops
- **Balanced Base Decomposition**  
  Represent elements in base-`b` with digits in `(-b/2, b/2]`
- **Norm Computation**  
  ℓ₂ and ℓ∞ norms with bound checking
- **Johnson–Lindenstrauss (JL) Projection**  
  Randomized projection with reproducible seeds
- **Random Vector Sampling**  
  Efficient, seedable generation of vectors over `Zq` or `Rq`
- **Challenge Sampling**  
  Rejection sampling of polynomials satisfying operator norm bounds

For example, the **labrador** protocol builds on this foundation to implement a lattice-based zk-SNARK with modular components and device acceleration.

## [See a full Rust example here.](https://github.com/ingonyama-zk/icicle/tree/main/examples/rust/lattice-snarks)

## Core Types

### Integer Ring: Zq

The integer ring `Zq` represents integers modulo `q`, where `q` is typically a product of small prime fields for efficiency.

The modulus q used in this library is a special 64-bit prime constructed as the product of two 32-bit primes:

```
q = P_babybear × P_koalabear
  = 0x78000001 × 0x7f000001
  = 0x3b880000f7000001
  = 4289678649214369793
```

#### Example

```rust
use icicle_core::{bignum::BigNum, traits::GenerateRandom};
use icicle_babykoala::ring::ScalarRing as Zq;

// Generate random Zq elements
let size = 100;
let zq_random: Vec<Zq> = Zq::generate_random(size);

// Generate zeros Zq elements
let zq_zeros: Vec<Zq> = vec![Zq::default(); size];

// Generate elements from arbitrary bytes
let element_size = std::mem::size_of::<Zq>();
let some_bytes: Vec<u8> = vec![0; element_size * size];
let zq_from_bytes: Vec<Zq> = some_bytes
    .chunks(element_size)
    .map(Zq::from_bytes_le)
    .collect();
```

### Polynomial Ring: Rq

The polynomial ring `Rq = Zq[X]/(X^d + 1)` represents polynomials of degree less than `d` with coefficients in `Zq`.

#### Example

```rust
use icicle_core::{polynomial_ring::PolynomialRing, traits::GenerateRandom}; // traits
use icicle_babykoala::polynomial_ring::PolyRing as Rq; // concrete type
use icicle_babykoala::ring::ScalarRing as Zq; // concrete type
use icicle_runtime::IcicleError;

// Generate random polynomials
let size = 8;
let rq_random: Vec<Rq> = Rq::generate_random(size);
let rq_zeros: Vec<Rq> = vec![Rq::default(); size];

// Convert Zq chunks to Rq polynomials
let zq_zeros: Vec<Zq> = vec![Zq::default(); size * Rq::DEGREE];
let unwrap = |result: Result<_, IcicleError>| result.unwrap();
let rq_from_slice: Vec<Rq> = zq_zeros
    .chunks(Rq::DEGREE)
    .map(Rq::from_slice)
    .map(unwrap)
    .collect();

// Or from arbitrary bytes
```

### Reinterpreting Rq and Zq slices

Many ICICLE APIs are defined over scalar rings like Zq, but can be applied to polynomial ring vectors (Rq) by flattening the polynomials into a contiguous Zq slice. This is useful for operations like JL projection.

To enable this, ICICLE provides utilities to reinterpret slices of polynomials as slices of their base field elements, using the HostOrDeviceSlice trait abstraction.

```rust
/// # Source
/// [`icicle_runtime::memory`]
/// 
/// Reinterprets a slice of polynomials as a flat slice of their base field elements.
///
/// This enables treating `&[P]` (e.g. Rq) as `&[P::Base]` (e.g. Zq) for scalar operations.
///
/// # Safety
/// - `P` must be `#[repr(C)]` and match the layout of `[P::Base; DEGREE]`.
/// - Memory must be properly aligned and valid for reading.
#[inline(always)]
pub fn flatten_polyring_slice<'a, P>(
    input: &'a (impl HostOrDeviceSlice<P> + ?Sized),
) -> impl HostOrDeviceSlice<P::Base> + 'a
where
    P: PolynomialRing,
    P::Base: 'a,
{
    unsafe { reinterpret_slice::<P, P::Base>(input).expect("Invalid slice cast") }
}
```

```rust
/// # Source
/// [`icicle_runtime::memory`]
/// 
/// Reinterprets a mutable slice of polynomials as a flat mutable slice of their base field elements.
///
/// # Safety
/// - Layout must match `[P::Base; DEGREE]` exactly.
/// - Caller must ensure exclusive access and proper alignment.
#[inline(always)]
pub fn flatten_polyring_slice_mut<'a, P>(
    input: &'a mut (impl HostOrDeviceSlice<P> + ?Sized),
) -> impl HostOrDeviceSlice<P::Base> + 'a
where
    P: PolynomialRing,
    P::Base: 'a,
{
    unsafe { reinterpret_slice_mut::<P, P::Base>(input).expect("Invalid slice cast") }
}
```

:::note
These helpers use the general **reinterpret_slice** utility, which reinterprets memory across types when their sizes and alignments match.
:::

#### Example

```rust
use icicle_core::polynomial_ring::flatten_polyring_slice; // or flatten_polyring_slice_mut
use icicle_core::traits::GenerateRandom;
use icicle_babykoala::polynomial_ring::PolyRing as Rq;
use icicle_runtime::memory::HostSlice; // concrete type

let polynomials = Rq::generate_random(5);
let poly_slice = HostSlice::from_slice(&polynomials);

// Flatten into a Zq slice (5 × DEGREE elements)
let scalar_slice = flatten_polyring_slice(poly_slice);

// This can now be passed into scalar-only APIs like `jl_projection`, `check_norm_bound`, etc.
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

### NTT API

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
) -> Result<(), IcicleError>;
```

### Inplace NTT API

```rust

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
) -> Result<(), IcicleError> {
    P::ntt_inplace(inout, dir, cfg)
}
```

### Example

```rust
use icicle_core::negacyclic_ntt::{
    ntt_inplace,         // In-place NTT wrapper
    NTTDir,              // Transform direction (Forward or Inverse)
    NegacyclicNttConfig, // Configuration for backend/device/async
};
use icicle_core::traits::GenerateRandom;
use icicle_babykoala::polynomial_ring::PolyRing;
use icicle_runtime::memory::{DeviceVec, HostSlice};

// Generate random input on the host
let size = 16;
let input = PolyRing::generate_random(size);

// Allocate and transfer to device memory
let mut device_input = DeviceVec::<PolyRing>::device_malloc(size).expect("malloc failed");
device_input
    .copy_from_host(HostSlice::from_slice(&input))
    .expect("copy failed");

// Compute in-place (or out of place)
let config = NegacyclicNttConfig::default();
ntt_inplace(&mut device_input, NTTDir::kForward, &config).expect("ntt failed"); 
```

## Matrix Operations

### Matrix Multiplication and Transpose over Ring Elements

ICICLE provides generic APIs for performing matrix operations over Polynomial rings.

Supported operations include:

- Dense matrix multiplication (row-major layout)
- Matrix transposition (row-major input)

These are useful for vector dot-products, Ajtai-style commitments, and other algebraic primitives in lattice-based SNARKs.

### Main Imports

```rust
use icicle_core::matrix_ops::{
    matmul,                 // Matrix multiplication
    matrix_transpose,       // Matrix transpose
    MatMulConfig,           // Backend and execution configuration
    MatrixOps,              // Trait for matmul    
};
use icicle_core::vec_ops::VecOpsConfig;
```

### Matrix multiplication API

```rust
/// Computes C = A × B for two row-major matrices, with optional transposition of A and/or B.
///
/// - `a`: Input matrix A, shape (a_rows × a_cols). Treated as Aᵗ if `cfg.a_transposed` is true.
/// - `b`: Input matrix B, shape (b_rows × b_cols). Treated as Bᵗ if `cfg.b_transposed` is true.
/// - `cfg`: Execution configuration (e.g., transposition flags, memory location, async mode).
/// - `result`: Output buffer for matrix C, with shape:
///     - `(a_rows × b_cols)` if `cfg.result_transposed == false`
///     - `(b_cols × a_rows)` if `cfg.result_transposed == true`
///
/// Returns an error if the matrix dimensions are incompatible or the configuration is invalid.
pub fn matmul<T>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    a_rows: u32,
    a_cols: u32,
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    b_rows: u32,
    b_cols: u32,
    cfg: &MatMulConfig,
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError>
where
    T: MatrixOps<T>;
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
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError>
where
    T: MatrixOps<T>;
```

### Example

```rust
    use icicle_core::traits::GenerateRandom;
    use icicle_core::{
        matrix_ops::{matmul, matrix_transpose, MatMulConfig},
        vec_ops::VecOpsConfig,
    };
    use icicle_babykoala::polynomial_ring::PolyRing;
    use icicle_runtime::memory::{DeviceVec, HostSlice};

    let n = 8;
    let m = 64;

    // Generate a random matrix A ∈ [n × m] on the host (row-major layout)
    let host_a = PolyRing::generate_random((n * m) as usize);

    // Allocate device buffer for Aᵗ ∈ [m × n]
    let mut device_a_transposed =
        DeviceVec::<PolyRing>::device_malloc((n * m) as usize).expect("Failed to allocate transpose output");

    // Transpose Aᵗ = transpose(A) (from host memory to device memory)
    matrix_transpose(
        HostSlice::from_slice(&host_a),
        n,
        m,
        &VecOpsConfig::default(),
        &mut device_a_transposed,
    )
    .expect("Transpose failed");

    // Allocate output buffer for (Aᵗ)A ∈ [m × m]
    let mut device_a_transposed_a =
        DeviceVec::<PolyRing>::device_malloc((m * m) as usize).expect("Failed to allocate output matrix");

    // Compute (Aᵗ)A
    // Note that one matrix is on host memory and the other on device memory
    matmul(
        &device_a_transposed,
        m,
        n,
        HostSlice::from_slice(&host_a),
        n,
        m,
        &MatMulConfig::default(),
        &mut device_a_transposed_a,
    )
    .expect("Matmul failed");

    // Compute (Aᵗ)A fused (transpose fused to matmul)
    let mut cfg = MatMulConfig::default();
    cfg.a_transposed = true;
    matmul(
        HostSlice::from_slice(&host_a),
        n,
        m,
        HostSlice::from_slice(&host_a),
        n,
        m,
        &cfg,
        &mut device_a_transposed_a,
    )
    .expect("Matmul failed");
```

## Polynomial Ring Vector Operations

ICICLE provides efficient vector operations over polynomial ring slices (e.g., Rq, Tq).
These operations are defined generically for any type implementing the PolynomialRing trait and operate on buffers that implement the HostOrDeviceSlice trait abstraction.

### Supported VecOps

- **polyvec_add** – Elementwise addition: out[i] = a[i] + b[i]
- **polyvec_sub** – Elementwise subtraction: out[i] = a[i] - b[i]
- **polyvec_mul** – Elementwise multiplication (supported only for NTT domain Tq)
- **polyvec_mul_by_scalar** – Multiply each polynomial by a corresponding Zq scalar
- **polyvec_sum_reduce** – Sum all polynomials into a single output: out = Σᵢ a[i]

### API

```rust
/// Multiply each polynomial by its corresponding scalar
pub fn polyvec_mul_by_scalar<P>(
    input_polyvec: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_scalarvec: &(impl HostOrDeviceSlice<P::Base> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    P: PolynomialRing,
    P::Base: VecOps<P::Base>;
```

```rust
/// Elementwise multiply two vectors of polynomials (valid for NTT form `Tq`)
pub fn polyvec_mul<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    P: PolynomialRing,
    P::Base: VecOps<P::Base>;
```

```rust
/// Elementwise addition: result[i] = a[i] + b[i]
pub fn polyvec_add<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    P: PolynomialRing,
    P::Base: VecOps<P::Base>;
```

```rust
/// Elementwise subtraction: result[i] = a[i] - b[i]
pub fn polyvec_sub<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    P: PolynomialRing,
    P::Base: VecOps<P::Base>;
```

```rust
/// Reduce a vector to a single polynomial: result[0] = sum(a)
pub fn polyvec_sum_reduce<P>(
    input_polyvec: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    P: PolynomialRing,
    P::Base: VecOps<P::Base>;
```

### Example

```rust
use icicle_core::polynomial_ring::PolynomialRing;
use icicle_core::traits::GenerateRandom;
use icicle_core::vec_ops::{
    poly_vecops::{polyvec_mul_by_scalar, polyvec_sum_reduce},
    VecOpsConfig,
};
use icicle_babykoala::polynomial_ring::PolyRing as Rq;
use icicle_babykoala::ring::ScalarRing as Zq;
use icicle_runtime::memory::{DeviceVec, HostSlice};

let size = 10;

// Generate a random vector of Zq scalars and a vector of Rq polynomials
let scalars = Zq::generate_random(size);
let polynomials = Rq::generate_random(size);

// Allocate device memory for the output of scalar × polynomial multiplication
let mut scaled_polynomials = DeviceVec::<Rq>::device_malloc(size).expect("Failed to allocate device memory");

// Perform elementwise multiplication: result[i] = scalars[i] × polynomials[i]
polyvec_mul_by_scalar(
    HostSlice::from_slice(&polynomials),
    HostSlice::from_slice(&scalars),
    &mut scaled_polynomials,
    &VecOpsConfig::default(),
)
.expect("polyvec_mul_by_scalar failed");

// Allocate a single Rq element to hold the sum-reduced result
let mut reduced = vec![Rq::zero(); 1];

// Reduce the vector of polynomials to a single polynomial by summation
polyvec_sum_reduce(
    &scaled_polynomials,
    HostSlice::from_mut_slice(&mut reduced),
    &VecOpsConfig::default(),
)
.expect("polyvec_sum_reduce failed");
```

## Balanced Base Decomposition

### Decompose and Recompose Ring Elements

Balanced base decomposition expresses each ring element (e.g. `Zq`, `Rq`) as a sequence of digits in a given base `b`, where each digit lies in the interval `(-b/2, b/2]`.

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
pub fn count_digits<T: BalancedDecomposition<T>>(base: u32) -> u32
```

```rust
/// Decomposes a slice of elements into balanced base-`b` digits (digit-major layout).
///
/// - `input.len()` = number of elements to decompose
/// - `output.len()` must be `input.len() × num_digits`, where `num_digits ∈ [1, count_digits(base)]`
///
/// Digits are written in order of increasing significance, grouped by digit index.
pub fn decompose<T: BalancedDecomposition<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    T: BalancedDecomposition<T>;
```

```rust
/// Recomposes original elements from digit-major base-`b` decomposition.
///
/// - `input.len()` must be `output.len() × num_digits`, where `num_digits ∈ [1, count_digits(base)]`
///
/// Recomposition is exact only if all omitted higher-order digits were zero.
pub fn recompose<T: BalancedDecomposition<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    T: BalancedDecomposition<T>;
```

### Examples

```rust
use icicle_core::balanced_decomposition;
use icicle_core::traits::GenerateRandom;
use icicle_core::vec_ops::VecOpsConfig;
use icicle_babykoala::polynomial_ring::PolyRing as Rq;
use icicle_runtime::memory::{DeviceVec, HostSlice};

let base = 4; // Typically set to q^(1/t) for small t (e.g., t = 2, 4, 6)
let size = 1024;
// Compute number of digits per element for the given base
let digits = balanced_decomposition::count_digits::<Rq>(base);
let output_len = size * digits as usize;

// Generate input vector
let input = Rq::generate_random(size);

// Allocate device memory for digit-major output
let mut decomposed = DeviceVec::<Rq>::device_malloc(output_len).expect("Failed to allocate device memory");

// Perform balanced base decomposition
balanced_decomposition::decompose::<Rq>(
    HostSlice::from_slice(&input),
    &mut decomposed,
    base,
    &VecOpsConfig::default(),
)
.expect("Decomposition failed");
```

## Norm Bound Checking

ICICLE provides an API to check whether the norm of a `Zq` vector is within a specified bound.

The API supports:

- ℓ₂ norm checking (sum of squares)
- ℓ∞ norm checking (maximum absolute value)
- Batch support

### Main Imports

```rust
use icicle_core::norm::{
    check_norm_bound, // Public wrapper function
    NormType,          // Enum to select ℓ₂ or ℓ∞ norm
    Norm,              // Trait implemented per field type
};
```

### API

```rust
pub enum NormType {
    /// ℓ₂ norm: sqrt(sum of squares)
    L2,
    /// ℓ∞ norm: max absolute value
    LInfinity,
}
```

```rust
/// Checks whether the norm of a vector (or batch of vectors) is within a given bound.
///
/// - `input`: Input slice of field elements (`Zq`)
/// - `norm_type`: `L2` or `LInfinity`
/// - `norm_bound`: The norm upper bound
/// - `cfg`: execution configuration
/// - `output`: Boolean results per batch
///
/// Interpretation:
/// - If `output.len() == 1`, checks the full input vector
/// - If `output.len() == B`, input is treated as `B` contiguous vectors (input.len() must be divisible by B)
pub fn check_norm_bound<T: IntegerRing>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    norm_type: NormType,
    norm_bound: u64,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<bool> + ?Sized),
) -> Result<(), IcicleError>
where
    T: Norm<T>;
```

### Example

```rust
use icicle_core::{bignum::BigNum, norm, vec_ops::VecOpsConfig};
use icicle_babykoala::ring::ScalarRing as Zq;
use icicle_runtime::memory::HostSlice;

let size = 1024;
let batch = 4;
let bound = 1000;

let input: Vec<Zq> = (0..size)
    .map(Zq::from_u32)
    .collect();
let mut output = vec![false; batch];

let cfg = VecOpsConfig::default();

// Interpretation:
// If output has 4 elements, the input is split into 4 sub-vectors (256 each).
// Norm is computed per sub-vector.
norm::check_norm_bound(
    HostSlice::from_slice(&input),
    norm::NormType::L2, // or NormType::LInfinity
    bound,
    &cfg,
    HostSlice::from_mut_slice(&mut output),
)
.expect("Norm check failed");

// Output[i] == true indicates that sub-vector i passed the norm bound.
```

## Johnson–Lindenstrauss (JL) Projection

ICICLE provides APIs for performing Johnson–Lindenstrauss (JL) projections, which reduce high-dimensional vectors into lower-dimensional spaces using pseudo-random sparse matrices.

### Supported Capabilities

- Seed-based projection of a vector of `Zq` elements using a sparse matrix with values in `{−1, 0, 1}`
- Seed-based projection of a vector of `Rq` elements by reinterpreting them as `Zq` coefficients
- Querying JL projection matrix rows deterministically:
  - As raw `Zq` values (for verification)
  - As grouped `Rq` polynomials, with optional conjugation (`a(X) → a(X⁻¹) mod Xⁿ + 1`) — useful for proof systems that prove the projection is computed correctly.

### Main imports

```rust
use icicle_core::jl_projection::{
    jl_projection, 
    get_jl_matrix_rows,
    get_jl_matrix_rows_as_polyring,
    JLProjection,
    JLProjectionPolyRing
};
```

### API

```rust
/// Projects a scalar vector into a lower-dimensional space using a pseudo-random JL matrix.
///
/// - `input.len()` = original dimensionality
/// - `output_projection.len()` = target dimensionality
/// - Projection matrix is seeded deterministically from `seed`
pub fn jl_projection<T>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    seed: &[u8],
    cfg: &VecOpsConfig,
    output_projection: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError>
where
    T: IntegerRing,
    T: JLProjection<T>;
```

```rust
/// Retrieves raw JL matrix rows over the scalar ring `T` in row-major order.
///
/// - Output layout: row 0 | row 1 | ... | row `num_rows - 1`
/// - Each row contains `row_size` scalar elements
pub fn get_jl_matrix_rows<T>(
    seed: &[u8],
    row_size: usize,
    start_row: usize,
    num_rows: usize,
    cfg: &VecOpsConfig,
    output_rows: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError>
where
    T: IntegerRing,
    T: JLProjection<T>;
```

```rust
/// Retrieves JL matrix rows as `Rq` polynomials, optionally conjugated.
///
/// - Each row contains `row_size` polynomials of degree `P::DEGREE`
/// - If `conjugate = true`, applies a(X) ↦ a(X⁻¹) mod Xⁿ + 1 to each polynomial
/// - Output is laid out row-major: row 0 | row 1 | ...
pub fn get_jl_matrix_rows_as_polyring<P>(
    seed: &[u8],
    row_size: usize,
    start_row: usize,
    num_rows: usize,
    conjugate: bool,
    cfg: &VecOpsConfig,
    output_rows: &mut (impl HostOrDeviceSlice<P> + ?Sized),
) -> Result<(), IcicleError>
where
    P: PolynomialRing + JLProjectionPolyRing<P>;
```

### Example

```rust
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{
    bignum::BigNum,
    jl_projection,
    polynomial_ring::{flatten_polyring_slice, PolynomialRing},
    traits::GenerateRandom,
};
use icicle_babykoala::polynomial_ring::PolyRing as Rq;
use icicle_babykoala::ring::ScalarRing as Zq;
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice};
use rand::Rng;

// Project a vector of Rq polyonmials
// Projecting 128 Rq polynomials (each of degree 64) to a 256-element Zq vector
let polynomials = Rq::generate_random(128);
let mut projection = vec![Zq::zero(); 256];

// Flatten the polynomials into a contiguous Zq slice (128 × 64 elements)
let flat_polynomials_as_zq = flatten_polyring_slice(HostSlice::from_slice(&polynomials));

// Use a random seed (e.g., hash of transcript or Fiat–Shamir challenge)
let mut seed = [0u8; 32];
rand::thread_rng().fill(&mut seed);

// Perform JL projection
jl_projection::jl_projection(
    &flat_polynomials_as_zq,
    &seed,
    &VecOpsConfig::default(),
    HostSlice::from_mut_slice(&mut projection),
)
.expect("JL projection failed");

// -----------------------------------------------------------------------------
// 🔍 Matrix Inspection (Zq form)
// -----------------------------------------------------------------------------

// Retrieve the first row of the JL matrix as Zq elements
let row_size = flat_polynomials_as_zq.len(); // same as input dimension
let mut output_rows = vec![Zq::zero(); row_size]; // 1 row × row_size elements

jl_projection::get_jl_matrix_rows(
    &seed,
    row_size, // row size (input dimension)
    0,        // start_row
    1,        // number of rows
    &VecOpsConfig::default(),
    HostSlice::from_mut_slice(&mut output_rows),
)
.expect("Failed to generate JL matrix rows as Zq");

// -----------------------------------------------------------------------------
// 🔁 Matrix Inspection (Rq form, with conjugation)
// -----------------------------------------------------------------------------

let mut output_rows_as_poly = vec![Rq::zero(); polynomials.len()]; // 1 row of polynomials

jl_projection::get_jl_matrix_rows_as_polyring(
    &seed,
    polynomials.len(), // row size (number of polynomials per row)
    0,                 // start_row
    1,                 // number of rows
    true,              // apply polynomial conjugation
    &VecOpsConfig::default(),
    HostSlice::from_mut_slice(&mut output_rows_as_poly),
)
.expect("Failed to generate JL matrix rows as Rq (conjugated)");
```

## Seeded Random Sampling

ICICLE provides an API for pseudorandom sampling of Zq and Rq elements on the device. This is useful for generating secret vectors, noise, or challenge polynomials in zero-knowledge protocols.

### Main imports

```rust
use icicle_core::random_sampling;
```

### API

```rust
/// Randomly samples elements of type `T` from a seeded uniform distribution.
///
/// This function fills the `output` buffer with pseudorandom elements of type `T`,
/// using the given `seed`.
/// 
/// # Parameters
/// - `fast_mode`:  Whether to use fast (non-cryptographic) sampling or secure sampling.
/// - `seed`:  byte slice used to deterministically seed the pseudorandom generator.
/// - `cfg`: execution configuration
/// - `output`:  Output buffer to store sampled elements
pub fn random_sampling<T>(
    fast_mode: bool,
    seed: &[u8],
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError>
where    
    T: RandomSampling<T>;
```

### Example

```rust
use icicle_core::{polynomial_ring::flatten_polyring_slice_mut, random_sampling, vec_ops::VecOpsConfig};
use icicle_babykoala::polynomial_ring::PolyRing as Rq;
use icicle_babykoala::ring::ScalarRing as Zq;
use icicle_runtime::memory::DeviceVec;
use rand::RngCore;

let size = 4096;
let fast_mode = true;
let cfg = VecOpsConfig::default();

// Generate a non-zero 32-byte seed for deterministic sampling
let mut seed = [0u8; 32];
rand::thread_rng().fill_bytes(&mut seed);

// --- Sample Zq elements ---
let mut zq_output = DeviceVec::<Zq>::device_malloc(size).expect("Zq alloc failed");
random_sampling::random_sampling(fast_mode, &seed, &cfg, &mut zq_output).expect("Zq sampling failed");

// --- Sample Rq polynomials ---
let mut rq_output = DeviceVec::<Rq>::device_malloc(size).expect("Rq alloc failed");
{
    // This scope is not necessary but we prefer to explicitly end the lifetime of the rq_as_zq reference
    let mut rq_as_zq = flatten_polyring_slice_mut(&mut rq_output);
    random_sampling::random_sampling(fast_mode, &seed, &cfg, &mut rq_as_zq).expect("Rq sampling failed");
}
```

## Challenge Sampling with Operator Norm Rejection

ICICLE provides a specialized API to sample challenge polynomials from a constrained subset of `Rq` that meet strict norm bounds. This is particularly relevant for lattice-based SNARK protocols like **labrador**.

The challenge space consists of Rq polynomials with:

- A fixed number of coefficients equal to ±1 (tau1)
- A fixed number of coefficients equal to ±2 (tau2)
- All remaining coefficients set to 0

The resulting polynomial is accepted only if it satisfies:

- An L-opnorm (operator norm) bound

Sampling is **deterministic** and based on a seed and output index. Polynomials exceeding the bound are rejected, and retries are performed deterministically to ensure reproducibility across devices and backends.

### Main imports

```rust
use icicle_core::random_sampling::{
    challenge_space_polynomials_sampling,  // Sampling function    
};
```

### API

```rust
/// Samples `Rq` challenge polynomials with coefficients in {0, ±1, ±2}.
///
/// This function generates polynomials from a constrained challenge space:
/// 1. Initializes each polynomial with `ones` coefficients set to ±1
///    and `twos` coefficients set to ±2 (randomly signed).
/// 2. Applies a random permutation to the coefficients.
/// 3. If `norm > 0`, applies operator norm rejection: only polynomials
///    with operator norm ≤ `norm` are accepted.
///
/// Sampling is deterministic based on the seed and internal indexing.
/// The output is a flat slice of polynomials (e.g., `&mut [T]` where `T: PolynomialRing`).
pub fn challenge_space_polynomials_sampling<T>(
    seed: &[u8],
    cfg: &VecOpsConfig,
    ones: usize,
    twos: usize,
    norm: usize,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError>
where
    T: PolynomialRing,
    T: ChallengeSpacePolynomialsSampling<T>;
```

### Example

```rust
use icicle_core::random_sampling::challenge_space_polynomials_sampling;
use icicle_core::vec_ops::VecOpsConfig;
use icicle_babykoala::polynomial_ring::PolyRing as Rq;
use icicle_runtime::memory::DeviceVec;
use rand::RngCore;

// Parameters from the labrador protocol
let tau1 = 31;           // Number of ±1 coefficients
let tau2 = 10;           // Number of ±2 coefficients
let opnorm_bound = 15;   // Operator norm bound for rejection sampling
let num_polynomials = 16;

// Generate a non-zero 60-byte deterministic seed (any seed size is valid)
let mut seed = [0u8; 60];
rand::thread_rng().fill_bytes(&mut seed);

// Allocate device memory for the output polynomials
let mut output = DeviceVec::<Rq>::device_malloc(num_polynomials)
    .expect("Failed to allocate device memory");

// Sample challenge polynomials with norm-based rejection
challenge_space_polynomials_sampling(
    &seed,
    &VecOpsConfig::default(),
    tau1,
    tau2,
    opnorm_bound, // Set to 0 to skip norm filtering
    &mut output,
)
.expect("Challenge space sampling failed");
```
