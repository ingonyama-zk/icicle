# Polynomial API Overview

:::note
Read our paper on the Polynomials API in ICICLE v2 by clicking [here](https://eprint.iacr.org/2024/973).
:::

## Introduction

The Polynomial API offers a robust framework for polynomial operations within a computational environment. It's designed for flexibility and efficiency, supporting a broad range of operations like arithmetic, evaluation, and manipulation, all while abstracting from the computation and storage specifics. This enables adaptability to various backend technologies, employing modern C++ practices.

## Key Features

### Backend Agnostic Architecture

Our API is structured to be independent of any specific computational backend. While a CUDA backend is currently implemented, the architecture facilitates easy integration of additional backends. This capability allows users to perform polynomial operations without the need to tailor their code to specific hardware, enhancing code portability and scalability.

### Templating in the Polynomial API

The Polynomial API is designed with a templated structure to accommodate different data types for coefficients, the domain, and images. This flexibility allows the API to be adapted for various computational needs and types of data.

```cpp
template <typename Coeff, typename Domain = Coeff, typename Image = Coeff>
class Polynomial {
    // Polynomial class definition
}
```

In this template:

- **`Coeff`**: Represents the type of the coefficients of the polynomial.
- **`Domain`**: Specifies the type for the input values over which the polynomial is evaluated. By default, it is the same as the type of the coefficients but can be specified separately to accommodate different computational contexts.
- **`Image`**: Defines the type of the output values of the polynomial. This is typically the same as the coefficients.

#### Default instantiation

```cpp
extern template class Polynomial<scalar_t>;
```

#### Extended use cases

The templated nature of the Polynomial API also supports more complex scenarios. For example, coefficients and images could be points on an elliptic curve (EC points), which are useful in cryptographic applications and advanced algebraic structures. This approach allows the API to be extended easily to support new algebraic constructions without modifying the core implementation.

### Supported Operations

The Polynomial class encapsulates a polynomial, providing a variety of operations:

- **Construction**: Create polynomials from coefficients or evaluations on roots-of-unity domains.
- **Arithmetic Operations**: Perform addition, subtraction, multiplication, and division.
- **Evaluation**: Directly evaluate polynomials at specific points or across a domain.
- **Manipulation**: Features like slicing polynomials, adding or subtracting monomials inplace, and computing polynomial degrees.
- **Memory Access**: Access internal states or obtain device-memory views of polynomials.

## Polynomial API Improvements

Since v2.3.0, ICICLE includes various fixes and performance enhancements for the Polynomial API, making it more robust and efficient for polynomial operations.

### Example: Polynomial API Improvements in C++
```cpp
#include <icicle/polynomial.h>

void improved_polynomial() {
    icicle::Polynomial p;
    p.coefficients = {4, 5, 6}; // p(x) = 6x^2 + 5x + 4
    p.print();
}
```

### Explanation
This example illustrates how to define and print a polynomial using the improved Polynomial API. The coefficients are set, and the polynomial is printed to the console.

## Usage

This section outlines how to use the Polynomial API in C++. Bindings for Rust and Go are detailed under the Bindings sections.

### Backend Initialization

Initialization with an appropriate factory is required to configure the computational context and backend.

```cpp
#include "polynomials/polynomials.h"
#include "polynomials/cuda_backend/polynomial_cuda_backend.cuh"

// Initialize with a CUDA backend
Polynomial::initialize(std::make_shared<CUDAPolynomialFactory>());
```

:::note
Initialization of a factory must be done per linked curve or field.
:::

### Construction

Polynomials can be constructed from coefficients, from evaluations on roots-of-unity domains, or by cloning existing polynomials.

```cpp
// Construction
static Polynomial from_coefficients(const Coeff* coefficients, uint64_t nof_coefficients);
static Polynomial from_rou_evaluations(const Image* evaluations, uint64_t nof_evaluations);
// Clone the polynomial
Polynomial clone() const;
```

Example:

```cpp
auto p_from_coeffs = Polynomial_t::from_coefficients(coeff /* :scalar_t* */, nof_coeffs);
auto p_from_rou_evals = Polynomial_t::from_rou_evaluations(rou_evals /* :scalar_t* */, nof_evals);
auto p_cloned = p.clone(); // p_cloned and p do not share memory
```

:::note
The coefficients or evaluations may be allocated either on host or device memory. In both cases the memory is copied to the backend device.
:::

### Arithmetic

Constructed polynomials can be used for various arithmetic operations:

```cpp
// Addition
Polynomial operator+(const Polynomial& rhs) const; 
Polynomial& operator+=(const Polynomial& rhs); // inplace addition

// Subtraction
Polynomial operator-(const Polynomial& rhs) const;

// Multiplication
Polynomial operator*(const Polynomial& rhs) const;
Polynomial operator*(const Domain& scalar) const; // scalar multiplication

// Division A(x) = B(x)Q(x) + R(x)
std::pair<Polynomial, Polynomial> divide(const Polynomial& rhs) const; // returns (Q(x), R(x))
Polynomial operator/(const Polynomial& rhs) const; // returns quotient Q(x)
Polynomial operator%(const Polynomial& rhs) const; // returns remainder R(x)
Polynomial divide_by_vanishing_polynomial(uint64_t degree) const; // sdivision by the vanishing polynomial V(x)=X^N-1
```

#### Example

Given polynomials A(x),B(x),C(x) and V(x) the vanishing polynomial.

$$
H(x)=\frac{A(x) \cdot B(x) - C(x)}{V(x)} \space where \space V(x) = X^{N}-1
$$

```cpp
auto H = (A*B-C).divide_by_vanishing_polynomial(N);
```

### Evaluation

Evaluate polynomials at arbitrary domain points, across a domain or on a roots-of-unity domain.

```cpp
Image operator()(const Domain& x) const; // evaluate f(x)
void evaluate(const Domain* x, Image* evals /*OUT*/) const;
void evaluate_on_domain(Domain* domain, uint64_t size, Image* evals /*OUT*/) const; // caller allocates memory
void evaluate_on_rou_domain(uint64_t domain_log_size, Image* evals /*OUT*/) const;  // caller allocate memory
```

Example:

```cpp
Coeff x = rand();
Image f_x = f(x); // evaluate f at x

// evaluate f(x) on a domain
uint64_t domain_size = ...;
auto domain = /*build domain*/; // host or device memory
auto evaluations = std::make_unique<scalar_t[]>(domain_size); // can be device memory too
f.evaluate_on_domain(domain, domain_size, evaluations);

// evaluate f(x) on roots of unity domain
uint64_t domain_log_size = ...;
auto evaluations_rou_domain = std::make_unique<scalar_t[]>(1 << domain_log_size); // can be device memory too
f.evaluate_on_rou_domain(domain_log_size, evaluations_rou_domain);
```

### Manipulations

Beyond arithmetic, the API supports efficient polynomial manipulations:

#### Monomials

```cpp
// Monomial operations
Polynomial& add_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);
Polynomial& sub_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);
```

The ability to add or subtract monomials directly and in-place is an efficient way to manipualte polynomials.

Example:

```cpp
f.add_monomial_in_place(scalar_t::from(5)); // f(x) += 5
f.sub_monomial_in_place(scalar_t::from(3), 8); // f(x) -= 3x^8
```

#### Computing the degree of a Polynomial

```cpp
// Degree computation
int64_t degree();
```

The degree of a polynomial is a fundamental characteristic that describes the highest power of the variable in the polynomial expression with a non-zero coefficient.
The `degree()` function in the API returns the degree of the polynomial, corresponding to the highest exponent with a non-zero coefficient.

- For the polynomial $f(x) = x^5 + 2x^3 + 4$, the degree is 5 because the highest power of $x$ with a non-zero coefficient is 5.
- For a scalar value such as a constant term (e.g., $f(x) = 7$, the degree is considered 0, as it corresponds to $x^0$.
- The degree of the zero polynomial, $f(x) = 0$, where there are no non-zero coefficients, is defined as -1. This special case often represents an "empty" or undefined state in many mathematical contexts.

Example:

```cpp
auto f = /*some expression*/;
auto degree_of_f = f.degree();
```

#### Slicing

```cpp
// Slicing and selecting even or odd components.
Polynomial slice(uint64_t offset, uint64_t stride, uint64_t size = 0 /*0 means take all elements*/);
Polynomial even();
Polynomial odd();
```

The Polynomial API provides methods for slicing polynomials and selecting specific components, such as even or odd indexed terms. Slicing allows extracting specific sections of a polynomial based on an offset, stride, and size.

The following examples demonstrate folding a polynomial's even and odd parts and arbitrary slicing;

```cpp
// folding a polynomials even and odd parts with randomness
auto x = rand();
auto even = f.even();
auto odd = f.odd();
auto fold_poly = even + odd * x;

// arbitrary slicing (first quarter)
auto first_quarter = f.slice(0 /*offset*/, 1 /*stride*/, f.degree()/4 /*size*/);
```

### Memory access (copy/view)

Access to the polynomial's internal state can be vital for operations like commitment schemes or when more efficient custom operations are necessary. This can be done either by copying or viewing the polynomial

#### Copying

Copies the polynomial coefficients to either host or device allocated memory.

:::note
Copying to host memory is backend agnostic while copying to device memory requires the memory to be allocated on the corresponding backend.
:::

```cpp
Coeff get_coeff(uint64_t idx) const; // copy single coefficient to host
uint64_t copy_coeffs(Coeff* coeffs, uint64_t start_idx, uint64_t end_idx) const;
```

Example:

```cpp
auto coeffs_device = /*allocate CUDA or host memory*/
f.copy_coeffs(coeffs_device, 0/*start*/, f.degree());
  
MSMConfig cfg = msm::defaultMSMConfig();
cfg.are_points_on_device = true; // assuming copy to device memory
auto rv = msm::MSM(coeffs_device, points, msm_size, cfg, results);
```

#### Views

The Polynomial API supports efficient data handling through the use of memory views. These views provide direct access to the polynomial's internal state without the need to copy data. This feature is particularly useful for operations that require direct access to device memory, enhancing both performance and memory efficiency.

##### What is a Memory View?

A memory view is essentially a pointer to data stored in device memory. By providing a direct access pathway to the data, it eliminates the need for data duplication, thus conserving both time and system resources. This is especially beneficial in high-performance computing environments where data size and operation speed are critical factors.

##### Applications of Memory Views

Memory views are extremely versatile and can be employed in various computational contexts such as:

- **Commitments**: Views can be used to commit polynomial states in cryptographic schemes, such as Multi-Scalar Multiplications (MSM).
- **External Computations**: They allow external functions or algorithms to utilize the polynomial's data directly, facilitating operations outside the core polynomial API. This is useful for custom operations that are not covered by the API.

##### Obtaining and Using Views

To create and use views within the Polynomial API, functions are provided to obtain pointers to both coefficients and evaluation data. Here’s how they are generally structured:

```cpp
// Obtain a view of the polynomial's coefficients
std::tuple<IntegrityPointer<Coeff>, uint64_t /*size*/, uint64_t /*device_id*/> get_coefficients_view();
```

Example usage:

```cpp
auto [coeffs_view, size, device_id] = polynomial.get_coefficients_view();

// Use coeffs_view in a computational routine that requires direct access to polynomial coefficients
// Example: Passing the view to a GPU-accelerated function
gpu_accelerated_function(coeffs_view.get(),...);
```

##### Integrity-Pointer: Managing Memory Views

Within the Polynomial API, memory views are managed through a specialized tool called the Integrity-Pointer. This pointer type is designed to safeguard operations by monitoring the validity of the memory it points to. It can detect if the memory has been modified or released, thereby preventing unsafe access to stale or non-existent data.
The Integrity-Pointer not only acts as a regular pointer but also provides additional functionality to ensure the integrity of the data it references. Here are its key features:

```cpp
// Checks whether the pointer is still considered valid
bool isValid() const;

// Retrieves the raw pointer or nullptr if pointer is invalid
const T* get() const;

// Dereferences the pointer. Throws exception if the pointer is invalid.
const T& operator*() const;

//Provides access to the member of the pointed-to object. Throws exception if the pointer is invalid.
const T* operator->() const;
```

Consider the Following case:

```cpp
auto [coeff_view, size, device] = f.get_coefficients_view();

// Use the coefficients view to perform external operations
commit_to_polynomial(coeff_view.get(), size);

// Modification of the original polynomial
f += g; // Any operation that modifies 'f' potentially invalidates 'coeff_view'

// Check if the view is still valid before using it further
if (coeff_view.isValid()) {
    perform_additional_computation(coeff_view.get(), size);
} else {
    handle_invalid_data();
}
```



## Multi-GPU Support with CUDA Backend

The Polynomial API includes comprehensive support for multi-GPU environments, a crucial feature for leveraging the full computational power of systems equipped with multiple NVIDIA GPUs. This capability is part of the API's CUDA backend, which is designed to efficiently manage polynomial computations across different GPUs.

### Setting the CUDA Device

Like other components of the icicle framework, the Polynomial API allows explicit setting of the current CUDA device:

```cpp
cudaSetDevice(int deviceID);
```

This function sets the active CUDA device. All subsequent operations that allocate or deal with polynomial data will be performed on this device.

### Allocation Consistency

Polynomials are always allocated on the current CUDA device at the time of their creation. It is crucial to ensure that the device context is correctly set before initiating any operation that involves memory allocation:

```cpp
// Set the device before creating polynomials
cudaSetDevice(0);
Polynomial p1 = Polynomial::from_coefficients(coeffs, size);

cudaSetDevice(1);
Polynomial p2 = Polynomial::from_coefficients(coeffs, size);
```

### Matching Devices for Operations

When performing operations that result in the creation of new polynomials (such as addition or multiplication), it is imperative that both operands are on the same CUDA device. If the operands reside on different devices, an exception is thrown:

```cpp
// Ensure both operands are on the same device
cudaSetDevice(0);
auto p3 = p1 + p2; // Throws an exception if p1 and p2 are not on the same device
```

### Device-Agnostic Operations

Operations that do not involve the creation of new polynomials, such as computing the degree of a polynomial or performing in-place modifications, can be executed regardless of the current device setting:

```cpp
// 'degree' and in-place operations do not require device matching
int deg = p1.degree();
p1 += p2; // Valid if p1 and p2 are on the same device, throws otherwise
```

### Error Handling

The API is designed to throw exceptions if operations are attempted across polynomials that are not located on the same GPU. This ensures that all polynomial operations are performed consistently and without data integrity issues due to device mismatches.

### Best Practices

To maximize the performance and avoid runtime errors in a multi-GPU setup, always ensure that:

- The CUDA device is set correctly before polynomial allocation.
- Operations involving new polynomial creation are performed with operands on the same device.

By adhering to these guidelines, developers can effectively harness the power of multiple GPUs to handle large-scale polynomial computations efficiently.
