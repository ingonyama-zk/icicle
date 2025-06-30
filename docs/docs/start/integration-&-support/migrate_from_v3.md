# Migration from ICICLE v3 to v4

ICICLE v4 introduces a more intuitive and object-oriented API for field elements, making code more readable and maintainable. This guide will assist you in transitioning from ICICLE v3 to v4 by highlighting the key changes and providing examples for both C++ and Rust.

## Key Conceptual Changes

- **Object-Oriented Field API**: In v3, field operations were performed using static methods on the field type (e.g., `FF::sqr(a)`). In v4, these operations are now instance methods on the field elements themselves (e.g., `a.sqr()`), making the code more natural and readable.
  
- **Method Chaining**: The new API supports method chaining, allowing you to write more concise and expressive code by chaining multiple operations together.

## Migration Guide for C++

### Replacing Static Method Calls with Instance Methods

In ICICLE v4, static method calls on field types have been replaced with instance methods on field elements. This change makes the code more intuitive and follows object-oriented principles.

#### Field Arithmetic Operations

| v3 (Static Methods) | v4 (Instance Methods) |
|---------------------|----------------------|
| `FF::add(a, b)` | `a.add(b)` or `a + b` |
| `FF::sub(a, b)` | `a.sub(b)` or `a - b` |
| `FF::mul(a, b)` | `a.mul(b)` or `a * b` |
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

### Replacing Trait Implementation Calls with Instance Methods

In ICICLE v4, the Rust bindings have been updated to use instance methods on field elements instead of trait implementation calls.

#### Field Arithmetic Operations

| v3 (Trait Implementation) | v4 (Instance Methods) |
|---------------------------|----------------------|
| `F::add(a, b)` | `a.add(b)` or `a + b` |
| `F::sub(a, b)` | `a.sub(b)` or `a - b` |
| `F::mul(a, b)` | `a.mul(b)` or `a * b` |
| `F::sqr(a)` | `a.sqr()` |
| `F::inv(a)` | `a.inv()` |
| `F::pow(a, exp)` | `a.pow(exp)` |

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

## Migration Guide for Golang

### Replacing Function Calls with Instance Methods

In ICICLE v4, the Golang bindings have also been updated to use instance methods on field elements.

#### Field Arithmetic Operations

| v3 (Function Calls) | v4 (Instance Methods) |
|---------------------|----------------------|
| `Add(a, b)` | `a.Add(&b)` |
| `Sub(a, b)` | `a.Sub(&b)` |
| `Mul(a, b)` | `a.Mul(&b)` |
| `Sqr(a)` | `a.Sqr()` |
| `Inv(a)` | `a.Inv()` |

### Example Migration

**v3 (Function Calls):**
```go
import "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/bn254"

a := bn254.NewScalarField().FromUint32(5)
b := bn254.NewScalarField().FromUint32(10)
c := bn254.Add(a, b)
d := bn254.Mul(c, bn254.NewScalarField().FromUint32(2))
e := bn254.Sqr(d)
f := bn254.Inv(e)
```

**v4 (Instance Methods):**
```go
import "github.com/ingonyama-zk/icicle/v4/wrappers/golang/fields/bn254"

a := bn254.NewScalarField().FromUint32(5)
b := bn254.NewScalarField().FromUint32(10)
c := a.Add(&b)
d := c.Mul(bn254.NewScalarField().FromUint32(2))
e := d.Sqr()
f := e.Inv()
```

## Other Considerations

- **API Consistency**: The v4 API provides a more consistent experience across different programming languages, making it easier to switch between them.
- **Code Readability**: The new API improves code readability by making the operations more intuitive and reducing the need for explicit type references.
- **Performance**: These API changes are purely syntactic and do not affect the performance of the underlying operations.

For further details and examples, refer to the [Programmer's Guide](start/programmers_guide/general.md). 