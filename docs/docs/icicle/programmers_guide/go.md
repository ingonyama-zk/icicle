# ICICLE Golang Usage Guide

## Overview

This guide covers the usage of Icicle's Golang API, including device management, memory operations, data transfer, synchronization, and compute APIs.

## Device Management

:::note
See all ICICLE runtime APIs in [runtime.go](https://github.com/ingonyama-zk/icicle/blob/yshekel/V3/wrappers/golang/runtime/runtime.go)
:::

### Loading a Backend

The backend can be loaded from a specific path or from an environment variable. This is essential for setting up the computing environment.

```go
import "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

result := runtime.LoadBackendFromEnvOrDefault()
// or load from custom install dir
result := runtime.LoadBackend("/path/to/backend/installdir", true)
```

### Setting and Getting Active Device

You can set the active device for the current thread and retrieve it when needed:

```go
device = runtime.CreateDevice("CUDA", 0) // or other
result := runtime.SetDevice(device)
// or query current (thread) device 
activeDevice := runtime.GetActiveDevice()
```

### Querying Device Information

Retrieve the number of available devices and check if a pointer is allocated on the host or on the active device:

```go
numDevices := runtime.GetDeviceCount()

var ptr unsafe.Pointer
isHostMemory = runtime.IsHostMemory(ptr)
isDeviceMemory = runtime.IsActiveDeviceMemory(ptr)
```

## Memory Management

### Allocating and Freeing Memory

Memory can be allocated and freed on the active device:

```go
ptr, err := runtime.Malloc(1024) // Allocate 1024 bytes
err := runtime.Free(ptr) // Free the allocated memory
```

### Asynchronous Memory Operations

You can perform memory allocation and deallocation asynchronously using streams:

```go
stream, err := runtime.CreateStream()

ptr, err := runtime.MallocAsync(1024, stream)
err = runtime.FreeAsync(ptr, stream)
```

### Querying Available Memory

Retrieve the total and available memory on the active device:

```go
size_t total_memory, available_memory;
availableMemory, err := runtime.GetAvailableMemory()
freeMemory := availableMemory.Free
totalMemory := availableMemory.Total
```

### Setting Memory Values

Set memory to a specific value on the active device, synchronously or asynchronously:

```go
err := runtime.Memset(ptr, 0, 1024) // Set 1024 bytes to 0
err := runtime.MemsetAsync(ptr, 0, 1024, stream)
```

## Data Transfer

### Explicit Data Transfers

To avoid device-inference overhead, use explicit copy functions:

```go
result := runtime.CopyToHost(host_dst, device_src, size)
result := runtime.CopyToHostAsync(host_dst, device_src, size, stream)
result := runtime.CopyToDevice(device_dst, host_src, size)
result := runtime.CopyToDeviceAsync(device_dst, host_src, size, stream)
```

## Stream Management

### Creating and Destroying Streams

Streams are used to manage asynchronous operations:

```go
stream, err := runtime.CreateStream()
err = runtime.DestroyStream(stream)
```

## Synchronization

### Synchronizing Streams and Devices

Ensure all previous operations on a stream or device are completed before proceeding:

```go
err := runtime.StreamSynchronize(stream)
err := runtime.DeviceSynchronize()
```

## Device Properties

### Checking Device Availability

Check if a device is available and retrieve a list of registered devices:

```go
dev := runtime.CreateDevice("CPU", 0)
isCPUAvail := runtime.IsDeviceAvailable(dev)
```

### Querying Device Properties

Retrieve properties of the active device:

```go
properties, err := runtime.GetDeviceProperties(properties);

/******************/
// where DeviceProperties is
type DeviceProperties struct {
  UsingHostMemory      bool   // Indicates if the device uses host memory
  NumMemoryRegions     int32  // Number of memory regions available on the device
  SupportsPinnedMemory bool   // Indicates if the device supports pinned memory
}
```


TODO




## Compute APIs

### Multi-Scalar Multiplication (MSM) Example

Icicle provides high-performance compute APIs such as the Multi-Scalar Multiplication (MSM) for cryptographic operations. Here's a simple example of how to use the MSM API.

```cpp
#include <iostream>
#include "icicle/runtime.h"
#include "icicle/api/bn254.h"

using namespace bn254;

int main()
{
  // Load installed backends
  icicle_load_backend_from_env_or_default();

  // trying to choose CUDA if available, or fallback to CPU otherwise (default device)
  const bool is_cuda_device_available = (eIcicleError::SUCCESS == icicle_is_device_available("CUDA"));
  if (is_cuda_device_available) {
    Device device = {"CUDA", 0};             // GPU-0
    ICICLE_CHECK(icicle_set_device(device)); // ICICLE_CHECK asserts that the api call returns eIcicleError::SUCCESS
  } // else we stay on CPU backend

  // Setup inputs
  int msm_size = 1024;
  auto scalars = std::make_unique<scalar_t[]>(msm_size);
  auto points = std::make_unique<affine_t[]>(msm_size);
  projective_t result;

  // Generate random inputs
  scalar_t::rand_host_many(scalars.get(), msm_size);
  projective_t::rand_host_many(points.get(), msm_size);

  // (optional) copy scalars to device memory explicitly
  scalar_t* scalars_d = nullptr;
  auto err = icicle_malloc((void**)&scalars_d, sizeof(scalar_t) * msm_size);
  // Note: need to test err and make sure no errors occurred
  err = icicle_copy(scalars_d, scalars.get(), sizeof(scalar_t) * msm_size);

  // MSM configuration
  MSMConfig config = default_msm_config();
  // tell icicle that the scalars are on device. Note that EC points and result are on host memory in this example.
  config.are_scalars_on_device = true;

  // Execute the MSM kernel (on the current device)
  eIcicleError result_code = msm(scalars_d, points.get(), msm_size, config, &result);
  // OR call bn254_msm(scalars_d, points.get(), msm_size, config, &result);

  // Free the device memory
  icicle_free(scalars_d);

  // Check for errors
  if (result_code == eIcicleError::SUCCESS) {
    std::cout << "MSM result: " << projective_t::to_affine(result) << std::endl;
  } else {
    std::cerr << "MSM computation failed with error: " << get_error_string(result_code) << std::endl;
  }

  return 0;
}
```

### Polynomial Operations Example

Here's another example demonstrating polynomial operations using Icicle:

```cpp
#include <iostream>
#include "icicle/runtime.h"
#include "icicle/polynomials/polynomials.h"
#include "icicle/api/bn254.h"

using namespace bn254;

// define bn254Poly to be a polynomial over the scalar field of bn254
using bn254Poly = Polynomial<scalar_t>;

static bn254Poly randomize_polynomial(uint32_t size)
{
  auto coeff = std::make_unique<scalar_t[]>(size);
  for (int i = 0; i < size; i++)
    coeff[i] = scalar_t::rand_host();
  return bn254Poly::from_rou_evaluations(coeff.get(), size);
}

int main()
{
  // Load backend and set device
  icicle_load_backend_from_env_or_default();

  // trying to choose CUDA if available, or fallback to CPU otherwise (default device)
  const bool is_cuda_device_available = (eIcicleError::SUCCESS == icicle_is_device_available("CUDA"));
  if (is_cuda_device_available) {
    Device device = {"CUDA", 0};             // GPU-0
    ICICLE_CHECK(icicle_set_device(device)); // ICICLE_CHECK asserts that the API call returns eIcicleError::SUCCESS
  } // else we stay on CPU backend

  int poly_size = 1024;

  // build domain for ntt is required for some polynomial ops that rely on ntt
  ntt_init_domain(scalar_t::omega(12), default_ntt_init_domain_config());

  // randomize polynomials f(x),g(x) over the scalar field of bn254
  bn254Poly f = randomize_polynomial(poly_size);
  bn254Poly g = randomize_polynomial(poly_size);

  // Perform polynomial multiplication
  auto result = f * g; // Executes on the current device

  ICICLE_LOG_INFO << "Done";

  return 0;
}
```

In this example, the polynomial multiplication is used to perform polynomial multiplication on CUDA or CPU, showcasing the flexibility and power of Icicle's compute APIs.

## Error Handling

### Checking for Errors

Icicle APIs return an `eIcicleError` enumeration value. Always check the returned value to ensure that operations were successful.

```cpp
if (result != eIcicleError::SUCCESS) {
    // Handle error
}
```

This guide provides an overview of the essential APIs available in Icicle for C++. The provided examples should help you get started with integrating Icicle into your high-performance computing projects.
