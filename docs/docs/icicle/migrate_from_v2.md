
# Migration from Icicle V2 to V3

Icicle V3 introduces a unified interface for high-performance computing across various devices, extending the functionality that was previously limited to GPUs in Icicle V2. This guide will help you transition from Icicle V2 to V3 by highlighting the key changes and providing examples for both C++ and Rust.

## Key Conceptual Changes

- **Device Independence**: In V2, Icicle was primarily designed for GPU computation, utilizing CUDA APIs directly. In V3, Icicle has evolved to support any computational device, including CPUs, GPUs, and other accelerators. This means the CUDA APIs have been replaced with device-agnostic Icicle APIs.
  
- **Unified API**: The APIs are now standardized across all devices, ensuring consistency in usage and reducing the complexity of handling different hardware backends.

## Migration Guide for C++

### Replacing CUDA APIs with Icicle APIs

In Icicle V3, CUDA-specific APIs have been replaced with Icicle APIs, which are designed to be backend-agnostic. This change allows your code to run on different devices without modification.

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

- **API Names**: While most API names remain consistent, they are now part of a more generalized runtime that can support multiple devices. Ensure you update the crate imports and function calls accordingly.
- **Backend Loading**: Ensure that you are loading the appropriate backend using the `load_backend_from_env_or_default` function, which is essential for setting up the runtime environment.

For further details and examples, refer to the [Icicle Rust Programmer's Guide](#).

---

This migration guide should help you transition your existing Icicle V2 projects to V3 smoothly, ensuring that you can take full advantage of the new device-agnostic framework.
