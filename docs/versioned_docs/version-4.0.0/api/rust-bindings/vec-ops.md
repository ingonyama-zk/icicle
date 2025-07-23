# Vector Operations API

Our vector operations API includes fundamental methods for addition, subtraction, multiplication, division, and more, with support for both host and device memory, as well as batched operations.

## Vector Operations Configuration

The `VecOpsConfig` struct encapsulates the settings for vector operations, including device context, operation modes, and batching parameters.

### `VecOpsConfig`

Defines configuration parameters for vector operations.

```rust
pub struct VecOpsConfig {
    pub stream_handle: IcicleStreamHandle,
    pub is_a_on_device: bool,
    pub is_b_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub batch_size: i32,
    pub columns_batch: bool,
    pub ext: ConfigExtension,
}
```

#### Fields

- **`stream_handle: IcicleStreamHandle`**: Specifies the stream (queue) to use for async execution
- **`is_a_on_device: bool`**: Indicates whether the input data a has been preloaded on the device memory. If `false` inputs will be copied from host to device.
- **`is_b_on_device: bool`**: Indicates whether the input b data has been preloaded on the device memory. If `false` inputs will be copied from host to device.
- **`is_result_on_device: bool`**: Indicates whether the output data is preloaded in device memory. If `false` outputs will be copied from host to device.
- **`is_async: bool`**: Specifies whether the operation should be performed asynchronously.
- **`batch_size: i32`**: Number of vector operations to process in a single batch. Each operation will be performed independently on each batch element. It is implicitly determined given the inputs and outputs to the vector operation.
- **`columns_batch: bool`**: true if the batched vectors are stored as columns in a 2D array (i.e., the vectors are strided in memory as columns of a matrix). If false, the batched vectors are stored contiguously in memory (e.g., as rows or in a flat array). Default is false.
- **`ext: ConfigExtension`**: Extended configuration for backend. Default is `ConfigExtension::new()`.

### Default Configuration

`VecOpsConfig` can be initialized with default settings tailored for a specific device:

```rust
let cfg = VecOpsConfig::default();
```

## Vector Operations

Vector operations are implemented through the `VecOps` trait, providing methods for addition, subtraction, multiplication, division, inversion, reduction, and more. These methods support both single and batched operations based on the batch_size and columns_batch configurations.

### Methods

All operations are element-wise operations, and the results placed into the `result` param. These operations are not in place, except for accumulate and bit_reverse_inplace.

- **`add`**: Computes the element-wise sum of two vectors.
- **`accumulate`**: Sum input b to a inplace.
- **`sub`**: Computes the element-wise difference between two vectors.
- **`mul`**: Performs element-wise multiplication of two vectors.
- **`div`**: Performs element-wise division of two vectors.
- **`inv`**: Computes the element-wise inverse of a vector.
- **`sum`**: Reduces a vector to its sum (optionally batched).
- **`product`**: Reduces a vector to its product (optionally batched).
- **`scalar_add`**: Adds a scalar to each element of a vector (batched).
- **`scalar_sub`**: Subtracts a scalar from each element of a vector (batched).
- **`scalar_mul`**: Multiplies each element of a vector by a scalar (batched).
- **`bit_reverse/bit_reverse_inplace`**: Reverse order of elements based on bit-reverse.
- **`slice`**: Extracts a strided slice from a vector.

```rust
pub fn add_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn accumulate_scalars<F>(
    a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn sub_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn mul_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn div_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn inv_scalars<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn sum_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn product_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn scalar_add<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn scalar_sub<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn scalar_mul<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn bit_reverse<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), IcicleError>;

pub fn bit_reverse_inplace<F>(
    input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>;

pub fn slice<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    offset: u64,
    stride: u64,
    size_in: u64,
    size_out: u64,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), IcicleError>;
```