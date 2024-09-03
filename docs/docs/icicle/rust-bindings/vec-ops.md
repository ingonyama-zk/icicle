# Vector Operations API

Our vector operations API includes fundamental methods for addition, subtraction, and multiplication of vectors, with support for both host and device memory.

## Vector Operations Configuration

The `VecOpsConfig` struct encapsulates the settings for vector operations, including device context and operation modes.

### `VecOpsConfig`

Defines configuration parameters for vector operations.

```rust
pub struct VecOpsConfig {
    pub stream_handle: IcicleStreamHandle,
    pub is_a_on_device: bool,
    pub is_b_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}
```

#### Fields

- **`stream_handle: IcicleStreamHandle`**: Specifies the stream (queue) to use for async execution
- **`is_a_on_device: bool`**: Indicates whether the input data a has been preloaded on the device memory. If `false` inputs will be copied from host to device.
- **`is_b_on_device: bool`**: Indicates whether the input b data has been preloaded on the device memory. If `false` inputs will be copied from host to device.
- **`is_result_on_device: bool`**: Indicates whether the output data is preloaded in device memory. If `false` outputs will be copied from host to device.
- **`is_async: bool`**: Specifies whether the NTT operation should be performed asynchronously.
- **`ext: ConfigExtension`**: extended configuration for backend.

### Default Configuration

`VecOpsConfig` can be initialized with default settings tailored for a specific device:

```rust
let cfg = VecOpsConfig::default();
```

## Vector Operations

Vector operations are implemented through the `VecOps` trait, providing methods for addition, subtraction, and multiplication of vectors.

### Methods

All operations are element-wise operations, and the results placed into the `result` param. These operations are not in place.

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