
# Migration from Icicle V2 to V3

Icicle V3 introduces a unified interface for high-performance computing across various devices, extending the functionality that was previously limited to GPUs in Icicle V2. This guide will assist you in transitioning from Icicle V2 to V3 by highlighting the key changes and providing examples for both C++ and Rust.

## Key Conceptual Changes

- **Device Independence**: n V2, Icicle was primarily designed for GPU computation, directly utilizing CUDA APIs. In V3, Icicle has evolved to support a broader range of computational devices, including CPUs, GPUs, and other accelerators. As a result, CUDA APIs have been replaced with device-agnostic Icicle APIs.
  
- **Unified API**: The APIs are now standardized across all devices, ensuring consistent usage and reducing the complexity of managing different hardware backends.

:::warning
When migrating from V2 to V3, it is important to note that, by default, your code now executes on the CPU. This contrasts with V2, which was exclusively a CUDA library. For details on installing and using CUDA GPUs, refer to the [CUDA backend guide](./install_cuda_backend.md).
:::

## Migration Guide for C++

### Replacing CUDA APIs with Icicle APIs

In Icicle V3, CUDA-specific APIs have been replaced with Icicle APIs that are designed to be backend-agnostic. This allows your code to run on different devices without requiring modifications.

- **Device Management**: Use Icicle's device management APIs instead of CUDA-specific functions. For example, instead of `cudaSetDevice()`, you would use `icicle_set_device()`.

- **Memory Management**: Replace CUDA memory management functions such as `cudaMalloc()` and `cudaFree()` with Icicle's `icicle_malloc()` and `icicle_free()`.

- **Stream Management**: Replace `cudaStream_t` with `icicleStreamHandle` and use Icicle's stream management functions.

For a detailed overview and examples, please refer to the [Icicle C++ Programmer's Guide](./programmers_guide/cpp.md) for full API details.

### Example Migration

**V2 (CUDA-specific):**
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
void* device_ptr;
cudaMalloc(&device_ptr, 1024);
// Perform operations using CUDA APIs
cudaStreamDestroy(stream);
cudaFree(device_ptr);
```

**V3 (Device-agnostic):**
```cpp
icicleStreamHandle stream;
icicle_create_stream(&stream);
void* device_ptr;
icicle_malloc(&device_ptr, 1024);
// Perform operations using Icicle APIs
icicle_destroy_stream(stream);
icicle_free(device_ptr);
```

## Migration Guide for Rust

### Replacing `icicle_cuda_runtime` with `icicle_runtime`

In Icicle V3, the `icicle_cuda_runtime` crate is replaced with the `icicle_runtime` crate. This change reflects the broader support for different devices beyond just CUDA-enabled GPUs.

- **Device Management**: Use `icicle_runtime`'s device management functions instead of those in `icicle_cuda_runtime`. The `Device` struct remains central, but it's now part of a more generalized runtime.

- **Memory Abstraction**: The `DeviceOrHostSlice` trait remains for memory abstraction, allowing seamless data handling between host and device.

- **Stream Management**: Replace `CudaStream` with `IcicleStream`, which now supports broader device types.

### Example Migration

**V2 (`icicle_cuda_runtime`):**
```rust
use icicle_cuda_runtime::{CudaStream, DeviceVec, HostSlice};

let mut stream = CudaStream::create().unwrap();
let mut device_memory = DeviceVec::cuda_malloc(1024).unwrap();
// Perform operations using CudaStream and related APIs
stream.synchronize().unwrap();
```

**V3 (`icicle_runtime`):**
```rust
use icicle_runtime::{IcicleStream, DeviceVec, HostSlice};

let mut stream = IcicleStream::create().unwrap();
let mut device_memory = DeviceVec::device_malloc(1024).unwrap();
// Perform operations using IcicleStream and related APIs
stream.synchronize().unwrap();
```

### Other Considerations

- **API Names**: While most API names remain consistent, they are now part of a more generalized runtime that can support multiple devices. Ensure that you update the crate imports and function calls accordingly.
- **Backend Loading**: Ensure that you are loading the appropriate backend using the `load_backend_from_env_or_default()` function, which is essential for setting up the runtime environment.

For further details and examples, refer to the [Programmer's Guide](./programmers_guide/general.md).
