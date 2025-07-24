# Rust FFI Bindings for Univariate Polynomial

:::note
Please refer to the Polynomials overview page for a deep overview. This section is a brief description of the Rust FFI bindings.
:::

This documentation is designed to provide developers with a clear understanding of how to utilize the Rust bindings for polynomial operations efficiently and effectively, leveraging the robust capabilities of both Rust and C++ in their applications.

## Introduction

The Rust FFI bindings for the Univariate Polynomial serve as a "shallow wrapper" around the underlying C++ implementation. These bindings provide a straightforward Rust interface that directly calls functions from a C++ library, effectively bridging Rust and C++ operations. The Rust layer handles simple interface translations without delving into complex logic or data structures, which are managed on the C++ side. This design ensures efficient data handling, memory management, and execution of polynomial operations directly via C++.
Currently, these bindings are tailored specifically for polynomials where the coefficients, domain, and images are represented as scalar fields.

## Initialization Requirements

Before utilizing any functions from the polynomial API, it is mandatory to initialize the appropriate polynomial backend (e.g., CUDA). Additionally, the NTT (Number Theoretic Transform) domain must also be initialized, as the CUDA backend relies on this for certain operations. Failing to properly initialize these components can result in errors.

:::note
**Field-Specific Initialization Requirement**

The ICICLE library is structured such that each field or curve has its dedicated library implementation. As a result, initialization must be performed individually for each field or curve to ensure the correct setup and functionality of the library.
:::

## Core Trait: `UnivariatePolynomial`

The `UnivariatePolynomial` trait encapsulates the essential functionalities required for managing univariate polynomials in the Rust ecosystem. This trait standardizes the operations that can be performed on polynomials, regardless of the underlying implementation details. It allows for a unified approach to polynomial manipulation, providing a suite of methods that are fundamental to polynomial arithmetic.

### Trait Definition

```rust
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_core::ring::IntegerRing;

pub trait UnivariatePolynomial: Clone + Sized {
    type Coeff: IntegerRing;

    // Create a polynomial from coefficients (lowest-degree first).
    fn from_coeffs<S: HostOrDeviceSlice<Self::Coeff> + ?Sized>(coeffs: &S, size: usize) -> Self;

    // Create a polynomial from evaluations on a roots-of-unity (power-of-two) domain.
    fn from_rou_evals<S: HostOrDeviceSlice<Self::Coeff> + ?Sized>(evals: &S, size: usize) -> Self;

    // Polynomial long division.  Returns (quotient, remainder).
    fn divide(&self, denominator: &Self) -> (Self, Self) where Self: Sized;

    // Divide by the vanishing polynomial X^degree − 1.
    fn div_by_vanishing(&self, degree: u64) -> Self;

    // In-place monomial updates.
    fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Coeff, monomial: u64);
    fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Coeff, monomial: u64);

    // Slicing helpers.
    fn slice(&self, offset: u64, stride: u64, size: u64) -> Self;
    fn even(&self) -> Self;                // keep even-indexed terms only
    fn odd(&self) -> Self;                 // keep odd-indexed terms only

    // Evaluation helpers.
    fn eval(&self, x: &Self::Coeff) -> Self::Coeff;
    fn eval_on_domain<D: HostOrDeviceSlice<Self::Coeff> + ?Sized,
                      E: HostOrDeviceSlice<Self::Coeff> + ?Sized>(
        &self,
        domain: &D,
        evals: &mut E,
    );
    fn eval_on_rou_domain<E: HostOrDeviceSlice<Self::Coeff> + ?Sized>(
        &self,
        domain_log_size: u64,
        evals: &mut E,
    );

    // Misc accessors.
    fn get_coeff(&self, idx: u64) -> Self::Coeff;
    fn copy_coeffs<S: HostOrDeviceSlice<Self::Coeff> + ?Sized>(&self, start_idx: u64, coeffs: &mut S);
    fn degree(&self) -> i64;
}
```

> **Note**  All `HostOrDeviceSlice` parameters accept either CPU memory (`HostSlice`) or GPU memory (`DeviceSlice`/`DeviceVec`).  The implementation handles transfers automatically based on the `is_*_on_device` flags inside the various config structs.

### `DensePolynomial`

The concrete wrapper that implements `UnivariatePolynomial` is called `DensePolynomial`.  Its public API remains the same; only the generic parameter names in the docs below were updated from `F` to `Self::Coeff` for clarity.

```rust
pub struct DensePolynomial {
    handle: PolynomialHandle,
}
```

### Traits implementation and methods

#### `Drop`

Ensures proper resource management by releasing the CUDA memory when a DensePolynomial instance goes out of scope. This prevents memory leaks and ensures that resources are cleaned up correctly, adhering to Rust's RAII (Resource Acquisition Is Initialization) principles.

#### `Clone`

Provides a way to create a new instance of a DensePolynomial with its own unique handle, thus duplicating the polynomial data in the CUDA context. Cloning is essential since the DensePolynomial manages external resources, which cannot be safely shared across instances without explicit duplication.

#### Operator Overloading: `Add`, `Sub`, `Mul`, `Rem`, `Div`

These traits are implemented for references to DensePolynomial (i.e., &DensePolynomial), enabling natural mathematical operations such as addition (+), subtraction (-), multiplication (*), division (/), and remainder (%). This syntactic convenience allows users to compose complex polynomial expressions in a way that is both readable and expressive.

#### Key Methods

In addition to the traits, the following methods are implemented:

```rust
impl DensePolynomial {
    /// Returns a **mutable** slice of the coefficients that lives on the active device.
    pub fn coeffs_mut_slice(&mut self) -> &mut DeviceSlice<Self::Coeff> { ... }
}
```

## Flexible Memory Handling With `HostOrDeviceSlice`

The DensePolynomial API is designed to accommodate a wide range of computational environments by supporting both host and device memory through the `HostOrDeviceSlice` trait. This approach ensures that polynomial operations can be seamlessly executed regardless of where the data resides, making the API highly adaptable and efficient for various hardware configurations.

### Overview of `HostOrDeviceSlice`

The HostOrDeviceSlice is a Rust trait that abstracts over slices of memory that can either be on the host (CPU) or the device (GPU), as managed by CUDA. This abstraction is crucial for high-performance computing scenarios where data might need to be moved between different memory spaces depending on the operations being performed and the specific hardware capabilities available.

### Usage in API Functions

Functions within the DensePolynomial API that deal with polynomial coefficients or evaluations use the HostOrDeviceSlice trait to accept inputs. This design allows the functions to be agnostic of the actual memory location of the data, whether it's in standard system RAM accessible by the CPU or in GPU memory accessible by CUDA cores.

```rust
// Assume `coeffs` could either be in host memory or CUDA device memory
let coeffs: DeviceSlice<F> = DeviceVec::<F>::malloc(coeffs_len);
let p_from_coeffs = PolynomialBabyBear::from_coeffs(&coeffs, coeffs.len());

// Similarly for evaluations from roots of unity
let evals: HostSlice<F> = HostSlice::from_slice(&host_memory_evals);
let p_from_evals = PolynomialBabyBear::from_rou_evals(&evals, evals.len());

// Same applies for any API that accepts HostOrDeviceSlice
```

## Usage

This section outlines practical examples demonstrating how to utilize the `DensePolynomial` Rust API. The API is flexible, supporting multiple scalar fields. Below are examples showing how to use polynomials defined over different fields and perform a variety of operations.

### Initialization and Basic Operations

First, choose the appropriate field implementation for your polynomial operations, initializing the CUDA backend if necessary

```rust
use icicle_babybear::polynomials::DensePolynomial as PolynomialBabyBear;

let f = PolynomialBabyBear::from_coeffs(...);

// now use f by calling the implemented traits

// For operations over another field, such as BN254
use icicle_bn254::polynomials::DensePolynomial as PolynomialBn254;
// Use PolynomialBn254 similarly
```

### Creation

Polynomials can be created from coefficients or evaluations:

```rust
let coeffs = ...;
let p_from_coeffs = PolynomialBabyBear::from_coeffs(HostSlice::from_slice(&coeffs), size);

let evals = ...;
let p_from_evals = PolynomialBabyBear::from_rou_evals(HostSlice::from_slice(&evals), size);

```

### Arithmetic Operations

Utilize overloaded operators for intuitive mathematical expressions:

```rust
let add = &f + &g;  // Addition
let sub = &f - &g;  // Subtraction
let mul = &f * &g;  // Multiplication
let mul_scalar = &f * &scalar;  // Scalar multiplication
```

### Division and Remainder

Compute quotient and remainder or perform division by a vanishing polynomial:

```rust
let (q, r) = f.divide(&g);  // Compute both quotient and remainder
let q = &f / &g;  // Quotient
let r = &f % &g;  // Remainder

let h = f.div_by_vanishing(N);  // Division by V(x) = X^N - 1

```

### Monomial Operations

Add or subtract monomials in-place for efficient polynomial manipulation:

```rust
f.add_monomial_inplace(&three, 1 /*monmoial*/); // Adds 3*x to f
f.sub_monomial_inplace(&one, 0 /*monmoial*/);   // Subtracts 1 from f
```

### Slicing

Extract specific components:

```rust
let even = f.even();  // Polynomial of even-indexed terms
let odd = f.odd();    // Polynomial of odd-indexed terms
let arbitrary_slice = f.slice(offset, stride, size);
```

### Evaluate

Evaluate the polynoomial:

```rust
let x = rand();  // Random field element
let f_x = f.eval(&x);  // Evaluate f at x

// Evaluate on a predefined domain
let domain = [one, two, three];
let mut host_evals = vec![ScalarField::zero(); domain.len()];
f.eval_on_domain(HostSlice::from_slice(&domain), HostSlice::from_mut_slice(&mut host_evals));

// Evaluate on roots-of-unity-domain
let domain_log_size = 4;
let mut device_evals = DeviceVec::<ScalarField>::malloc(1 << domain_log_size);
f.eval_on_rou_domain(domain_log_size, device_evals.into_slice_mut());
```

### Read coefficients

Read or copy polynomial coefficients for further processing:

```rust
let x_squared_coeff = f.get_coeff(2);  // Coefficient of x^2

// Copy coefficients to a device-specific memory space
let mut device_mem = DeviceVec::<Field>::malloc(coeffs.len());
f.copy_coeffs(0, device_mem.into_slice_mut());
```

### Polynomial Degree

Determine the highest power of the variable with a non-zero coefficient:

```rust
let deg = f.degree();  // Degree of the polynomial
```

### Memory Management: Views (rust slices)

Rust enforces correct usage of views at compile time, eliminating the need for runtime checks:

```rust
let mut f = Poly::from_coeffs(HostSlice::from_slice(&coeffs), size);

// Obtain a mutable slice of coefficients as a DeviceSlice
let coeffs_slice_dev = f.coeffs_mut_slice();

// Operations on f are restricted here due to mutable borrow of coeffs_slice_dev

// Compute evaluations or perform other operations directly using the slice
// example: evaluate f on a coset of roots-of-unity. Computing from GPU to HOST/GPU
let mut config: NTTConfig<'_, F> = NTTConfig::default();
config.coset_gen = /*some coset gen*/;
let mut coset_evals = vec![F::zero(); coeffs_slice_dev.len()];
ntt(
    coeffs_slice_dev,
    NTTDir::kForward,
    &config,
    HostSlice::from_mut_slice(&mut coset_evals),
)
.unwrap();

// now can f can be borrowed once again
```
