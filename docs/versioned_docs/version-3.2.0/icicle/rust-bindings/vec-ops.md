# Vector Operations API

Our vector operations API includes fundamental methods for addition, subtraction, and multiplication of vectors, with support for both host and device memory, as well as batched operations.

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
    pub batch_size: usize,
    pub columns_batch: bool,
    pub ext: ConfigExtension,
}
```

#### Fields

- **`stream_handle: IcicleStreamHandle`**: Specifies the stream (queue) to use for async execution
- **`is_a_on_device: bool`**: Indicates whether the input data a has been preloaded on the device memory. If `false` inputs will be copied from host to device.
- **`is_b_on_device: bool`**: Indicates whether the input b data has been preloaded on the device memory. If `false` inputs will be copied from host to device.
- **`is_result_on_device: bool`**: Indicates whether the output data is preloaded in device memory. If `false` outputs will be copied from host to device.
- **`is_async: bool`**: Specifies whether the NTT operation should be performed asynchronously.
- **`batch_size: usize`**: Number of vector operations to process in a single batch. Each operation will be performed independently on each batch element.
- **`columns_batch: bool`**: true if the batched vectors are stored as columns in a 2D array (i.e., the vectors are strided in memory as columns of a matrix). If false, the batched vectors are stored contiguously in memory (e.g., as rows or in a flat array).

- **`ext: ConfigExtension`**: extended configuration for backend.

### Default Configuration

`VecOpsConfig` can be initialized with default settings tailored for a specific device:

```rust
let cfg = VecOpsConfig::default();
```

## Vector Operations

Vector operations are implemented through the `VecOps` trait, providing methods for addition, subtraction, and multiplication of vectors. These methods support both single and batched operations based on the batch_size and columns_batch configurations.

### Methods

All operations are element-wise operations, and the results placed into the `result` param. These operations are not in place, except for accumulate.

- **`add`**: Computes the element-wise sum of two vectors.
- **`accumulate`**: Sum input b to a inplace.
- **`sub`**: Computes the element-wise difference between two vectors.
- **`mul`**: Performs element-wise multiplication of two vectors.
- **`transpose`**: Performs matrix transpose.
- **`bit_reverse/bit_reverse_inplace`**: Reverse order of elements based on bit-reverse.



```rust
pub fn add_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>;

pub fn accumulate_scalars<F>(
    a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>;

pub fn sub_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>;

pub fn mul_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>;

pub fn transpose_matrix<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    nof_rows: u32,
    nof_cols: u32,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>;

pub fn bit_reverse<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), eIcicleError>;

pub fn bit_reverse_inplace<F>(
    input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>;
```