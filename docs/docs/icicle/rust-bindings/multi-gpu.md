# Multi GPU APIs

To learn more about the theory of Multi GPU programming refer to [this part](../multi-gpu.md) of documentation.

Here we will cover the core multi GPU apis and a [example](#a-multi-gpu-example)


## A Multi GPU example

In this example we will display how you can

1. Fetch the number of devices installed on a machine
2. For every GPU launch a thread and set an active device per thread.
3. Execute a MSM on each GPU



```rust

...

let device_count = get_device_count().unwrap();

(0..device_count)
        .into_par_iter()
        .for_each(move |device_id| {
          set_device(device_id).unwrap();

          // you can allocate points and scalars_d here

          let mut cfg = MSMConfig::default_for_device(device_id);
          cfg.ctx.stream = &stream;
          cfg.is_async = true;
          cfg.are_scalars_montgomery_form = true;
          msm(&scalars_d, &HostOrDeviceSlice::on_host(points), &cfg, &mut msm_results).unwrap();

          // collect and process results
        })

...
```


We use `get_device_count` to fetch the number of connected devices, device IDs will be `0, 1, 2, ..., device_count - 1`

[`into_par_iter`](https://docs.rs/rayon/latest/rayon/iter/trait.IntoParallelIterator.html#tymethod.into_par_iter) is a parallel iterator, you should expect it to launch a thread for every iteration.

We then call `set_device(device_id).unwrap();` it should set the context of that thread to the selected `device_id`.

Any data you now allocate from the context of this thread will be linked to the `device_id`. We create our `MSMConfig` with the selected device ID `let mut cfg = MSMConfig::default_for_device(device_id);`, behind the scene this will create for us a `DeviceContext` configured for that specific GPU. 

We finally call our `msm` method.


## Device management API

To streamline device management we offer as part of `icicle-cuda-runtime` package methods for dealing with devices.

#### [`set_device`](https://github.com/ingonyama-zk/icicle/blob/e6035698b5e54632f2c44e600391352ccc11cad4/wrappers/rust/icicle-cuda-runtime/src/device.rs#L6)

Sets the current CUDA device by its ID, when calling `set_device` it will set the current thread to a CUDA device.

**Parameters:**

- `device_id: usize`: The ID of the device to set as the current device. Device IDs start from 0.

**Returns:**

- `CudaResult<()>`: An empty result indicating success if the device is set successfully. In case of failure, returns a `CudaError`.

**Errors:**

- Returns a `CudaError` if the specified device ID is invalid or if a CUDA-related error occurs during the operation.

**Example:**

```rust
let device_id = 0; // Device ID to set
match set_device(device_id) {
    Ok(()) => println!("Device set successfully."),
    Err(e) => eprintln!("Failed to set device: {:?}", e),
}
```

#### [`get_device_count`](https://github.com/ingonyama-zk/icicle/blob/e6035698b5e54632f2c44e600391352ccc11cad4/wrappers/rust/icicle-cuda-runtime/src/device.rs#L10)

Retrieves the number of CUDA devices available on the machine.

**Returns:**

- `CudaResult<usize>`: The number of available CUDA devices. On success, contains the count of CUDA devices. On failure, returns a `CudaError`.

**Errors:**

- Returns a `CudaError` if a CUDA-related error occurs during the retrieval of the device count.

**Example:**

```rust
match get_device_count() {
    Ok(count) => println!("Number of devices available: {}", count),
    Err(e) => eprintln!("Failed to get device count: {:?}", e),
}
```

#### [`get_device`](https://github.com/ingonyama-zk/icicle/blob/e6035698b5e54632f2c44e600391352ccc11cad4/wrappers/rust/icicle-cuda-runtime/src/device.rs#L15)

Retrieves the ID of the current CUDA device.

**Returns:**

- `CudaResult<usize>`: The ID of the current CUDA device. On success, contains the device ID. On failure, returns a `CudaError`.

**Errors:**

- Returns a `CudaError` if a CUDA-related error occurs during the retrieval of the current device ID.

**Example:**

```rust
match get_device() {
    Ok(device_id) => println!("Current device ID: {}", device_id),
    Err(e) => eprintln!("Failed to get current device: {:?}", e),
}
```

## Device context API

The `DeviceContext` is embedded into `NTTConfig`, `MSMConfig` and `PoseidonConfig`, meaning you can simply pass a `device_id` to your existing config and the same computation will be triggered on a different device.

#### [`DeviceContext`](https://github.com/ingonyama-zk/icicle/blob/e6035698b5e54632f2c44e600391352ccc11cad4/wrappers/rust/icicle-cuda-runtime/src/device_context.rs#L11)

Represents the configuration a CUDA device, encapsulating the device's stream, ID, and memory pool. The default device is always `0`.

```rust
pub struct DeviceContext<'a> {
    pub stream: &'a CudaStream,
    pub device_id: usize,
    pub mempool: CudaMemPool,
}
```

##### Fields

- **`stream: &'a CudaStream`**

  A reference to a `CudaStream`. This stream is used for executing CUDA operations. By default, it points to a null stream CUDA's default execution stream.

- **`device_id: usize`**

  The index of the GPU currently in use. The default value is `0`, indicating the first GPU in the system.

  In some cases assuming `CUDA_VISIBLE_DEVICES` was configured, for example as `CUDA_VISIBLE_DEVICES=2,3,7` in the system with 8 GPUs - the `device_id=0` will correspond to GPU with id 2. So the mapping may not always be a direct reflection of the number of GPUs installed on a system.

- **`mempool: CudaMemPool`**

  Represents the memory pool used for CUDA memory allocations. The default is set to a null pointer, which signifies the use of the default CUDA memory pool.

##### Implementation Notes

- The `DeviceContext` structure is cloneable and can be debugged, facilitating easier logging and duplication of contexts when needed.


#### [`DeviceContext::default_for_device(device_id: usize) -> DeviceContext<'static>`](https://github.com/ingonyama-zk/icicle/blob/e6035698b5e54632f2c44e600391352ccc11cad4/wrappers/rust/icicle-cuda-runtime/src/device_context.rs#L30)

Provides a default `DeviceContext` with system-wide defaults, ideal for straightforward setups.

#### Returns

A `DeviceContext` instance configured with:
- The default stream (`null_mut()`).
- The default device ID (`0`).
- The default memory pool (`null_mut()`).

#### Parameters

- **`device_id: usize`**: The ID of the device for which to create the context.

#### Returns

A `DeviceContext` instance with the provided `device_id` and default settings for the stream and memory pool.


#### [`check_device(device_id: i32)`](https://github.com/vhnatyk/icicle/blob/eef6876b037a6b0797464e7cdcf9c1ecfcf41808/wrappers/rust/icicle-cuda-runtime/src/device_context.rs#L42)

Validates that the specified `device_id` matches the ID of the currently active device, ensuring operations are targeted correctly.

#### Parameters

- **`device_id: i32`**: The device ID to verify against the currently active device.

#### Behavior

- **Panics** if the `device_id` does not match the active device's ID, preventing cross-device operation errors.

#### Example

```rust
let device_id: i32 = 0; // Example device ID
check_device(device_id);
// Ensures that the current context is correctly set for the specified device ID.
```
