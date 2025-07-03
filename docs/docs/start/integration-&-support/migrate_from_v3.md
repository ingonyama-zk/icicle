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
| `FF::neg(a)` | `a.neg()` or `-a` |
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

### Refactor Program from Vecops to Program module

In v4, the Program API has been moved from VecOps to a dedicated Program module for better organization and type safety.

| Operation | v3 (VecOps) | v4 (Program) |
|-----------|-------------|--------------|
| Create program | `Fr::create_program(|symbols| { ... }, nof_params)?` | `FieldProgram::new(|symbols| { ... }, nof_params)?` |
| Execute program | `Fr::execute_program(&program, &mut vec_data, &config)?` | `program.execute_program(&mut vec_data, &config)?` |
| Create returning value program | Not available | `FieldReturningValueProgram::new(|symbols| -> symbol { ... }, nof_params)?` |
| Create predefined program | `Fr::create_predefined_program(pre_def)?` | `FieldProgram::new_predefined(pre_def)?` |
| Imports | `use icicle_core::vec_ops::{VecOps, VecOpsConfig};` | `use icicle_core::program::{Program, ReturningValueProgram};` |

### Example Migration

**v3 (Trait Implementation):**
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

**v4 (Instance Methods):**
```rust
use icicle_fields::bn254::Fr;
use icicle_core::traits::FieldImpl;
use icicle_core::traits::Arithmetic;

let a = Fr::from_u32(5);
let b = Fr::from_u32(10);
let c = a + b;  // or a.add(b)
let d = c * Fr::from_u32(2);  // or c.mul(Fr::from_u32(2))
let e = d.sqr();
let f = e.inv();
```

**v4 (Method Chaining):**
```rust
use icicle_fields::bn254::Fr;
use icicle_core::traits::FieldImpl;
use icicle_core::traits::Arithmetic;

let result = Fr::from_u32(5)
    .add(Fr::from_u32(10))
    .mul(Fr::from_u32(2))
    .sqr()
    .inv();
```


## Other Considerations

- **API Consistency**: The v4 API provides a more consistent experience across different programming languages, making it easier to switch between them.
- **Code Readability**: The new API improves code readability by making the operations more intuitive and reducing the need for explicit type references.
- **Performance**: These API changes are purely syntactic and do not affect the performance of the underlying operations.

For further details and examples, refer to the [Programmer's Guide](start/programmers_guide/general.md).