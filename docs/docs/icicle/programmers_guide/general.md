
# Icicle Programmer's Guide

## General Concepts

### Compute APIs

Icicle offers a variety of compute APIs, including Number Theoretic Transforms (NTT), Multi Scalar Multiplication (MSM), vector operations, Elliptic Curve NTT (ECNTT), polynomials, and more. These APIs follow a consistent structure, making it straightforward to apply the same usage patterns across different operations.

#### Common Structure of Compute APIs

Each compute API in Icicle typically involves the following components:

- **Inputs and Outputs**: The data to be processed and the resulting output are passed to the API functions. These can reside either on the host (CPU) or on a device (GPU).

- **Parameters**: Parameters such as the size of data to be processed are provided to control the computation.

- **Configuration Struct**: A configuration struct is used to specify additional options for the computation. This struct has default values but can be customized as needed.

The configuration struct allows users to modify settings such as:

- Specifying whether inputs and outputs are on the host or device.
- Adjusting the data layout for specific optimizations.
- Passing custom options to the backend implementation through an extension mechanism, such as setting the number of CPU cores to use.

#### Example (C++)

```cpp
#include "icicle/vec_ops.h"

// vector_add is defined in vec_ops.h
template <typename T>
eIcicleError vector_add(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

// Create config struct for vector add
VecOpsConfig config = default_vec_ops_config();
// optionally modify the config struct

// Call the API
eIcicleError err = vector_add(vec_a, vec_b, size, config, vec_res);
```

Where `VecOpsConfig` is defined in `icicle/vec_ops.h`:

```cpp
struct VecOpsConfig {
    icicleStreamHandle stream; /**< Stream for asynchronous execution. */
    bool is_a_on_device;       /**< True if `a` is on the device, false if it is not. Default value: false. */
    bool is_b_on_device;       /**< True if `b` is on the device, false if it is not. Default value: false. OPTIONAL. */
    bool is_result_on_device;  /**< If true, the output is preserved on the device, otherwise on the host. Default value: false. */
    bool is_async;             /**< Whether to run the vector operations asynchronously. 
                                    If set to `true`, the function will be non-blocking and synchronization 
                                    must be explicitly managed using `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
                                    If set to `false`, the function will block the current CPU thread. */
    ConfigExtension* ext = nullptr; /**< Backend-specific extension. */
};
```

This pattern is consistent across most Icicle APIs, in C++/Rust/Go, providing flexibility while maintaining a familiar structure. For NTT, MSM, and other operations, include the corresponding header and call the template APIs.

:::note
High-level APIs, such as the polynomial API, further abstract these details, offering an even simpler interface for complex operations.
:::

:::note
Rather than using the template APIs, you can also use specialized APIs by including `icicle/api/babybear.h` (or any other) to avoid confusion when using multiple fields/curves.
:::

### Device Abstraction

Icicle provides a device abstraction layer that allows you to interact with different compute devices such as CPUs and GPUs seamlessly. The device abstraction ensures that your code can work across multiple hardware platforms without modification.

#### Device Management

- **Loading Backends**: Backends are loaded dynamically based on the environment configuration or a specified path.
- **Setting Active Device**: The active device for a thread can be set, allowing for targeted computation on a specific device.

### Streams

Streams in Icicle allow for asynchronous execution and memory operations, enabling parallelism and non-blocking execution. Streams are associated with specific devices, and you can create, destroy, and synchronize streams to manage your workflow.

:::note
For compute APIs, streams go into the `config.stream` field along with the `is_async=true` config flag.
:::

### Memory Management

Icicle provides functions for allocating, freeing, and managing memory across devices. Memory operations can be performed synchronously or asynchronously, depending on the use case.

### Data Transfer

Data transfer between the host and devices, or between different devices, is handled through a set of APIs that ensure efficient and error-checked operations. Asynchronous operations are supported to maximize performance.

### Synchronization

Synchronization ensures that all previous operations on a stream or device are completed. This is crucial when coordinating between multiple operations that depend on one another.

## Additional Information

- **Error Handling**: Icicle uses a specific error enumeration (`eIcicleError`) to handle and return error states across its APIs.
- **Device Properties**: You can query various properties of devices to tailor operations according to the hardware capabilities.
