# Vector Operations API

Our vector operations API which is part of `icicle-cuda-runtime` package, includes fundamental methods for addition, subtraction, and multiplication of vectors, with support for both host and device memory.

## Examples

### Addition of Scalars

```rust
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::vec_ops::{add_scalars};

let test_size = 1 << 18;

let a: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let b: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let mut result: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);

let cfg = VecOpsConfig::default();
add_scalars(&a, &b, &mut result, &cfg).unwrap();
```

### Subtraction of Scalars

```rust
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::vec_ops::{sub_scalars};

let test_size = 1 << 18;

let a: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let b: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let mut result: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);

let cfg = VecOpsConfig::default();
sub_scalars(&a, &b, &mut result, &cfg).unwrap();
```

### Multiplication of Scalars

```rust
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::vec_ops::{mul_scalars};

let test_size = 1 << 18;

let a: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let ones: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::one(); test_size]);
let mut result: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);

let cfg = VecOpsConfig::default();
mul_scalars(&a, &ones, &mut result, &cfg).unwrap();
```

## Vector Operations Configuration

The `VecOpsConfig` struct encapsulates the settings for vector operations, including device context and operation modes.

### `VecOpsConfig`

Defines configuration parameters for vector operations.

```rust
pub struct VecOpsConfig<'a> {
    pub ctx: DeviceContext<'a>,
    is_a_on_device: bool,
    is_b_on_device: bool,
    is_result_on_device: bool,
    pub is_async: bool,
}
```

#### Fields

- **`ctx: DeviceContext<'a>`**: Specifies the device context for the operation, including the device ID and memory pool.
- **`is_a_on_device`**: Indicates if the first operand vector resides in device memory.
- **`is_b_on_device`**: Indicates if the second operand vector resides in device memory.
- **`is_result_on_device`**: Specifies if the result vector should be stored in device memory.
- **`is_async`**: Enables asynchronous operation. If `true`, operations are non-blocking; otherwise, they block the current thread.

### Default Configuration

`VecOpsConfig` can be initialized with default settings tailored for a specific device:

```rust
let cfg = VecOpsConfig::default();
```

These are the default settings.

```rust
impl<'a> Default for VecOpsConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> VecOpsConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        VecOpsConfig {
            ctx: DeviceContext::default_for_device(device_id),
            is_a_on_device: false,
            is_b_on_device: false,
            is_result_on_device: false,
            is_async: false,
        }
    }
}
```

## Vector Operations

Vector operations are implemented through the `VecOps` trait, providing methods for addition, subtraction, and multiplication of vectors.

### `VecOps` Trait

```rust
pub trait VecOps<F> {
    fn add(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn sub(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn mul(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;
}
```

#### Methods

All operations are element-wise operations, and the results placed into the `result` param. These operations are not in place.

- **`add`**: Computes the element-wise sum of two vectors.
- **`sub`**: Computes the element-wise difference between two vectors.
- **`mul`**: Performs element-wise multiplication of two vectors.

## MatrixTranspose API Documentation

This section describes the functionality of the `TransposeMatrix` function used for matrix transposition.

The function takes a matrix represented as a 1D slice and transposes it, storing the result in another 1D slice.

### Function

```rust
pub fn transpose_matrix<F>(
    input: &HostOrDeviceSlice<F>,
    row_size: u32,
    column_size: u32,
    output: &mut HostOrDeviceSlice<F>,
    ctx: &DeviceContext,
    on_device: bool,
    is_async: bool,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>
```

### Parameters

- **`input`**: A slice representing the input matrix. The slice can be stored on either the host or the device.
- **`row_size`**: The number of rows in the input matrix.
- **`column_size`**: The number of columns in the input matrix.
- **`output`**: A mutable slice to store the transposed matrix. The slice can be stored on either the host or the device.
- **`ctx`**: A reference to the `DeviceContext`, which provides information about the device where the operation will be performed.
- **`on_device`**: A boolean flag indicating whether the inputs and outputs are on the device.
- **`is_async`**: A boolean flag indicating whether the operation should be performed asynchronously.

### Return Value

`Ok(())` if the operation is successful, or an `IcicleResult` error otherwise.

### Example

```rust
use icicle::HostOrDeviceSlice;
use icicle::DeviceContext;
use icicle::FieldImpl;
use icicle::VecOps;

let input: HostOrDeviceSlice<i32> = // ...;
let mut output: HostOrDeviceSlice<i32> = // ...;
let ctx: DeviceContext = // ...;

transpose_matrix(&input, 5, 4, &mut output, &ctx, true, false)
    .expect("Failed to transpose matrix");
```

The function takes a matrix represented as a 1D slice, transposes it, and stores the result in another 1D slice. The input and output slices can be stored on either the host or the device, and the operation can be performed synchronously or asynchronously.

The function is generic and can work with any type `F` that implements the `FieldImpl` trait. The `<F as FieldImpl>::Config` type must also implement the `VecOps<F>` trait, which provides the `transpose` method used to perform the actual transposition.

The function returns an `IcicleResult<()>`, indicating whether the operation was successful or not.
