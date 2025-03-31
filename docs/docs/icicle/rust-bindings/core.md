# Core Traits and Types

This document describes the core traits and types used throughout the ICICLE Rust bindings.

## Field Traits

### FieldImpl

The `FieldImpl` trait defines the basic interface for finite field elements:

```rust
pub trait FieldImpl: Display + Debug + PartialEq + Copy + Clone + Into<Self::Repr> + From<Self::Repr> + Send + Sync {
    type Config: FieldConfig;
    type Repr;

    fn to_bytes_le(&self) -> Vec<u8>;
    fn from_bytes_le(bytes: &[u8]) -> Self;
    fn from_hex(s: &str) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u32(val: u32) -> Self;
}
```

#### Methods

- **`to_bytes_le()`** - Converts the field element to a vector of bytes in little-endian order
- **`from_bytes_le()`** - Creates a field element from a slice of bytes in little-endian order
- **`from_hex()`** - Creates a field element from a hexadecimal string
- **`zero()`** - Returns the zero element of the field
- **`one()`** - Returns the multiplicative identity element of the field
- **`from_u32()`** - Creates a field element from a u32 value

### FieldConfig

The `FieldConfig` trait provides configuration for field implementations:

```rust
pub trait FieldConfig: Debug + PartialEq + Copy + Clone {
    fn from_u32<const NUM_LIMBS: usize>(val: u32) -> [u32; NUM_LIMBS];
}
```

#### Methods

- **`from_u32()`** - Converts a u32 value into an array of limbs with the specified size

### Arithmetic

The `Arithmetic` trait provides basic arithmetic operations for field elements:

```rust
pub trait Arithmetic: Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn sqr(self) -> Self;
    fn inv(self) -> Self;
    fn pow(self, exp: usize) -> Self;
}
```

#### Methods

- **`sqr()`** - Computes the square of the field element
- **`inv()`** - Computes the multiplicative inverse of the field element
- **`pow()`** - Raises the field element to the specified power

### MontgomeryConvertible

The `MontgomeryConvertible` trait provides methods for converting values to and from Montgomery form:

```rust
pub trait MontgomeryConvertible: Sized {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError;
    fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError;
}
```

#### Methods

- **`to_mont()`** - Converts values to Montgomery form using the provided stream
- **`from_mont()`** - Converts values from Montgomery form using the provided stream

## Curve Traits

### Curve

The `Curve` trait defines the interface for elliptic curve implementations:

```rust
pub trait Curve: Debug + PartialEq + Copy + Clone {
    type BaseField: FieldImpl;
    type ScalarField: FieldImpl;

    fn eq_proj(point1: *const Projective<Self>, point2: *const Projective<Self>) -> bool;
    fn to_affine(point: *const Projective<Self>, point_aff: *mut Affine<Self>);
    fn generate_random_projective_points(size: usize) -> Vec<Projective<Self>>;
    fn generate_random_affine_points(size: usize) -> Vec<Affine<Self>>;
    fn convert_affine_montgomery(points: *mut Affine<Self>, len: usize, is_into: bool, stream: &IcicleStream) -> eIcicleError;
    fn convert_projective_montgomery(points: *mut Projective<Self>, len: usize, is_into: bool, stream: &IcicleStream) -> eIcicleError;
    fn add(point1: Projective<Self>, point2: Projective<Self>) -> Projective<Self>;
    fn sub(point1: Projective<Self>, point2: Projective<Self>) -> Projective<Self>;
    fn mul_scalar(point1: Projective<Self>, point2: Self::ScalarField) -> Projective<Self>;
}
```

#### Associated Types

- `BaseField` - The base field type used for curve point coordinates
- `ScalarField` - The scalar field type used for scalar multiplication

#### Methods

- **`eq_proj()`** - Compares two projective points for equality
- **`to_affine()`** - Converts a projective point to affine coordinates
- **`generate_random_projective_points()`** - Generates a vector of random projective points
- **`generate_random_affine_points()`** - Generates a vector of random affine points
- **`convert_affine_montgomery()`** - Converts affine points to/from Montgomery form
- **`convert_projective_montgomery()`** - Converts projective points to/from Montgomery form
- **`add()`** - Adds two projective points
- **`sub()`** - Subtracts two projective points
- **`mul_scalar()`** - Multiplies a projective point by a scalar

## Point Types

### Projective

The `Projective` struct represents a point on an elliptic curve in projective coordinates:

```rust
pub struct Projective<C: Curve> {
    pub x: C::BaseField,
    pub y: C::BaseField,
    pub z: C::BaseField,
}
```

#### Fields

- `x` - X coordinate in projective form
- `y` - Y coordinate in projective form
- `z` - Z coordinate in projective form

### Affine

The `Affine` struct represents a point on an elliptic curve in affine coordinates:

```rust
pub struct Affine<C: Curve> {
    pub x: C::BaseField,
    pub y: C::BaseField,
}
```

#### Fields

- `x` - X coordinate
- `y` - Y coordinate

## Field Type

### Field

The `Field` struct represents a finite field element with a fixed number of limbs:

```rust
pub struct Field<const NUM_LIMBS: usize, F: FieldConfig> {
    limbs: [u32; NUM_LIMBS],
    p: PhantomData<F>,
}
```

#### Type Parameters

- `NUM_LIMBS` - The number of 32-bit limbs used to represent the field element
- `F` - The field configuration type implementing `FieldConfig`

#### Fields

- `limbs` - The limbs representing the field element
- `p` - Phantom data to hold the field configuration type 