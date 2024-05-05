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
pub trait UnivariatePolynomial
where
    Self::Field: FieldImpl,
    Self::FieldConfig: FieldConfig,
{
    type Field: FieldImpl;
    type FieldConfig: FieldConfig;

    // Methods to create polynomials from coefficients or roots-of-unity evaluations.
    fn from_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(coeffs: &S, size: usize) -> Self;
    fn from_rou_evals<S: HostOrDeviceSlice<Self::Field> + ?Sized>(evals: &S, size: usize) -> Self;

    // Method to divide this polynomial by another, returning quotient and remainder.
    fn divide(&self, denominator: &Self) -> (Self, Self) where Self: Sized;

    // Method to divide this polynomial by the vanishing polynomial 'X^N-1'.
    fn div_by_vanishing(&self, degree: u64) -> Self;

    // Methods to add or subtract a monomial in-place.
    fn add_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);
    fn sub_monomial_inplace(&mut self, monomial_coeff: &Self::Field, monomial: u64);

    // Method to slice the polynomial, creating a sub-polynomial.
    fn slice(&self, offset: u64, stride: u64, size: u64) -> Self;

    // Methods to return new polynomials containing only the even or odd terms.
    fn even(&self) -> Self;
    fn odd(&self) -> Self;

    // Method to evaluate the polynomial at a given domain point.
    fn eval(&self, x: &Self::Field) -> Self::Field;

    // Method to evaluate the polynomial over a domain and store the results.
    fn eval_on_domain<D: HostOrDeviceSlice<Self::Field> + ?Sized, E: HostOrDeviceSlice<Self::Field> + ?Sized>(
        &self,
        domain: &D,
        evals: &mut E,
    );

    // Method to retrieve a coefficient at a specific index.
    fn get_coeff(&self, idx: u64) -> Self::Field;

    // Method to copy coefficients into a provided slice.
    fn copy_coeffs<S: HostOrDeviceSlice<Self::Field> + ?Sized>(&self, start_idx: u64, coeffs: &mut S);

    // Method to get the degree of the polynomial.
    fn degree(&self) -> i64;
}
```

## `DensePolynomial` Struct

The DensePolynomial struct represents a dense univariate polynomial in Rust, leveraging a handle to manage its underlying memory within the CUDA device context. This struct acts as a high-level abstraction over complex C++ memory management practices, facilitating the integration of high-performance polynomial operations through Rust's Foreign Function Interface (FFI) bindings.

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
    pub fn init_cuda_backend() -> bool {...}
    // Returns a mutable slice of the polynomial coefficients on the device
    pub fn coeffs_mut_slice(&mut self) -> &mut DeviceSlice<F> {...}
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
let coeffs: DeviceSlice<F> = DeviceVec::<F>::cuda_malloc(coeffs_len).unwrap();
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

// Initialize the CUDA backend for polynomial operations
PolynomialBabyBear::init_cuda_backend();
let f = PolynomialBabyBear::from_coeffs(...);

// now use f by calling the implemented traits

// For operations over another field, such as BN254
use icicle_bn254::polynomials::DensePolynomial as PolynomialBn254;
// Use PolynomialBn254 similarly
```

### Creation

Polynomials can be created from coefficients or evaluations:

```rust
// Assume F is the field type (e.g. icicle_bn254::curve::ScalarField or a type parameter)
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
```

### Read coefficients

Read or copy polynomial coefficients for further processing:

```rust
let x_squared_coeff = f.get_coeff(2);  // Coefficient of x^2

// Copy coefficients to a device-specific memory space
let mut device_mem = DeviceVec::<Field>::cuda_malloc(coeffs.len()).unwrap();
f.copy_coeffs(0, &mut device_mem[..]);
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
