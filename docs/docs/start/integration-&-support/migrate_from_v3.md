# Migration from ICICLE v3 to v4

ICICLE v4 introduces a more intuitive and object-oriented C++ API for field elements, making code more readable and maintainable. This guide will assist you in transitioning from ICICLE v3 to v4 by highlighting the key changes and providing examples for both C++ and Rust.

## Key Conceptual Changes

- **Object-Oriented Field API**: In v3, field operations were performed using static methods on the field type (e.g., `FF::sqr(a)`). In v4, these operations are now instance methods on the field elements themselves (e.g., `a.sqr()`), making the code more natural and readable.
  
- **Method Chaining**: The new API supports method chaining, allowing you to write more concise and expressive code by chaining multiple operations together.

## Migration Guide for C++

### Replacing Static Method Calls with Instance Methods

In ICICLE v4, static method calls on field types have been replaced with instance methods on field elements. This change makes the code more intuitive and follows object-oriented principles.

#### Field Arithmetic Operations

| v3 (Static Methods) | v4 (Instance Methods) |
|---------------------|----------------------|
| `FF::add(a, b)` | `a + b` |
| `FF::sub(a, b)` | `a - b` |
| `FF::mul(a, b)` | `a * b` |
| `FF::sqr(a)` | `a.sqr()` |
| `FF::neg(a)` | `a.neg()` |
| `FF::inverse(a)` | `a.inverse()` |
| `FF::pow(a, exp)` | `a.pow(exp)` |

#### Montgomery Conversion

| v3 (Static Methods) | v4 (Instance Methods) |
|---------------------|----------------------|
| `FF::to_montgomery(a)` | `a.to_montgomery()` |
| `FF::from_montgomery(a)` | `a.from_montgomery()` |

### Example Migration

**v3 (Static Methods):**
```cpp
FF a = FF::from(5);
FF b = FF::from(10);
FF c = FF::add(a, b);
FF d = FF::mul(c, FF::from(2));
FF e = FF::sqr(d);
FF f = FF::inverse(e);
```

**v4 (Instance Methods):**
```cpp
FF a = FF::from(5);
FF b = FF::from(10);
FF c = a.add(b);  // or a + b
FF d = c.mul(FF::from(2));  // or c * FF::from(2)
FF e = d.sqr();
FF f = e.inverse();
```

**v4 (Method Chaining):**
```cpp
FF result = FF::from(5)
    .add(FF::from(10))
    .mul(FF::from(2))
    .sqr()
    .inverse();
```

## Migration Guide for Rust

### Arithmetic API

ICICLE v4 implements field arithmetic operations through the `Arithmetic` trait and standard Rust operators. This makes the code more idiomatic and easier to read.

#### Field Arithmetic Operations

| (Static Methods) | (Instance Methods) | (Operators) |
|---------------------|----------------------|----------------|
| `Fr::add(a, b)` | `a.add(b)` | `a + b` |
| `Fr::sub(a, b)` | `a.sub(b)` | `a - b` |
| `Fr::mul(a, b)` | `a.mul(b)` | `a * b` |
| `Fr::sqr(a)` | `a.sqr()` | N/A |
| `Fr::neg(a)` | `a.neg()` | `-a` |
| `Fr::inv(a)` | `a.inv()` | N/A |
| `Fr::pow(a, exp)` | `a.pow(exp)` | N/A |

#### The Arithmetic Trait

The `Arithmetic` trait in ICICLE v4 is defined as follows:

```rust
pub trait Arithmetic: Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn sqr(&self) -> Self;
    fn pow(&self, exp: usize) -> Self;
}
```

This trait extends the standard Rust operators (`Add`, `Sub`, `Mul`) and adds specialized field operations like square, inverse, and exponentiation.

#### The Invertible Trait

Field inversion is now provided by the `Invertible` trait. This trait defines a single method, `inv`, which returns the multiplicative inverse of a field element.

```rust
pub trait Invertible: Sized {
    fn inv(&self) -> Self;
}
```

If the field element is zero, the function will return zero.

### Renaming of the `FieldImpl` Trait

`FieldImpl` was renamed into `Field`

### Refactor Program from Vecops to Program module

In v4, the Program API has been moved from VecOps to a dedicated Program module for better organization and type safety.

#### Create program
- **v3 (VecOps):** `Fr::create_program(|symbols| { ... }, nof_params)?`
- **v4 (Program):** `FieldProgram::new(|symbols| { ... }, nof_params)?`

#### Execute program
- **v3 (VecOps):** `Fr::execute_program(&program, &mut vec_data, &config)?`
- **v4 (Program):** `program.execute_program(&mut vec_data, &config)?`

#### Create returning value program
- **v4:** `ReturningValueProgram::new(|symbols| -> symbol { ... }, nof_params)?`

#### Create predefined program
- **v3 (VecOps):** `Fr::create_predefined_program(pre_def)?`
- **v4 (Program):** `FieldProgram::new_predefined(pre_def)?`

#### Imports
- **v3 (VecOps):** `use icicle_core::vec_ops::{VecOps, VecOpsConfig};`
- **v4 (Program):** `use icicle_core::program::{Program, ReturningValueProgram};`

### Example Migration

**v3 (Old API):**
```rust
use icicle_fields::bn254::Fr;
use icicle_core::traits::FieldImpl;
use icicle_core::field::FieldArithmetic;

let a = Fr::from_u32(5);
let b = Fr::from_u32(10);
let c = Fr::add(a, b);
let d = Fr::mul(c, Fr::from_u32(2));
let e = Fr::sqr(d);
let f = Fr::inv(e);
```

**v4 (Current API):**
```rust
use icicle_fields::bn254::ScalarField;
use icicle_core::traits::Arithmetic;

let a = ScalarField::from(5);
let b = ScalarField::from(10);
let c = a + b;  // or a.add(b)
let d = c * ScalarField::from(2);  // or c.mul(ScalarField::from(2))
let e = d.sqr();
let f = e.inv();
```

**v4 (Method Chaining):**
```rust
use icicle_fields::bn254::ScalarField;
use icicle_core::traits::Arithmetic;

let result = ScalarField::from(5)
    .add(ScalarField::from(10))
    .mul(ScalarField::from(2))
    .sqr()
    .inv();
```

### Program API Example

**v3 (VecOps API):**
```rust
use icicle_fields::bn254::Fr;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};

// Create program
let program = Fr::create_program(|symbols| {
    // Program logic here
}, nof_params)?;

// Execute program
Fr::execute_program(&program, &mut vec_data, &config)?;
```

**v4 (Program API):**
```rust
use icicle_fields::bn254::ScalarField;
use icicle_core::program::{Program, ReturningValueProgram};
use icicle_bn254::program::stark252::{FieldProgram, FieldReturningValueProgram};

// Create program
let program = FieldProgram::new(|symbols| {
    // Program logic here
}, nof_params)?;

// Execute program
program.execute_program(&mut vec_data, &config)?;

// Create returning value program
let returning_program = FieldReturningValueProgram::new(|symbols| -> symbol {
    // Program logic here
    result_symbol
}, nof_params)?;
```

### Random Number Generation

**v3:**
```rust
use icicle_fields::bn254::ScalarField;
use icicle_core::traits::FieldCfg;

let random_values = ScalarField::generate_random(size);
```

**v4:**
```rust
use icicle_fields::bn254::ScalarField;
use icicle_core::traits::GenerateRandom;

let random_values = ScalarField::generate_random(size);
```

### Removal of the `FieldCfg` Trait

`FieldCfg` previously exposed compile-time field parameters (modulus, root, etc.) **and** helper functions like `generate_random`.  In v4 these responsibilities are split:

* Concrete field types now publish constants directly (e.g., `ScalarField::MODULUS`).
* Random sampling moved to the standalone `GenerateRandom` trait.

Migration steps:

1. Replace `use icicle_core::traits::FieldCfg;` with `use icicle_core::traits::GenerateRandom;` when you only need random values.
2. Access constants directly from the field type:

```rust
// v3
let p = <Fr as FieldCfg>::MODULUS;

// v4
let p = ScalarField::MODULUS;
```

All constant names are unchanged, so the update is usually a simple search-and-replace.

For further details and examples, refer to the [Programmer's Guide](start/programmers_guide/general.md).