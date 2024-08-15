# Icicle C++ Usage Guide

## Overview

This guide covers the usage of Icicle's C++ API, including device management, memory operations, data transfer, synchronization, and compute APIs.

## Device Management

### Loading a Backend

The backend can be loaded from a specific path or from an environment variable. This is essential for setting up the computing environment.

```cpp
#include "icicle/runtime.h"

eIcicleError result = icicle_load_backend("/path/to/backend", true);
```

To load the backend from an environment variable or default directory:

```cpp
eIcicleError result = icicle_load_backend_from_env_or_default();
```

### Setting and Getting Active Device

You can set the active device for the current thread and retrieve it when needed:

```cpp
icicle::Device device = {"CUDA", 0}; // or other
eIcicleError result = icicle_set_device(device);

eIcicleError result = icicle_get_active_device(device);
```

### Querying Device Information

Retrieve the number of available devices and check if a pointer is allocated on the host or on the active device:

```cpp
int device_count;
eIcicleError result = icicle_get_device_count(device_count);

bool is_host_memory;
eIcicleError result = icicle_is_host_memory(ptr);

bool is_device_memory;
eIcicleError result = icicle_is_active_device_memory(ptr);
```

## Memory Management

### Allocating and Freeing Memory

Memory can be allocated and freed on the active device:

```cpp
void* ptr;
eIcicleError result = icicle_malloc(&ptr, 1024); // Allocate 1024 bytes

eIcicleError result = icicle_free(ptr); // Free the allocated memory
```

### Asynchronous Memory Operations

You can perform memory allocation and deallocation asynchronously using streams:

```cpp
icicleStreamHandle stream;
icicle_create_stream(&stream);

void* ptr;
eIcicleError result = icicle_malloc_async(&ptr, 1024, stream);

eIcicleError result = icicle_free_async(ptr, stream);
```

### Querying Available Memory

Retrieve the total and available memory on the active device:

```cpp
size_t total_memory, available_memory;
eIcicleError result = icicle_get_available_memory(total_memory, available_memory);
```

### Setting Memory Values

Set memory to a specific value on the active device, synchronously or asynchronously:

```cpp
eIcicleError result = icicle_memset(ptr, 0, 1024); // Set 1024 bytes to 0

eIcicleError result = icicle_memset_async(ptr, 0, 1024, stream);
```

## Data Transfer

### Copying Data

Data can be copied between host and device, or between devices. The location of the memory is inferred from the pointers:

```cpp
eIcicleError result = icicle_copy(dst, src, size);
eIcicleError result = icicle_copy_async(dst, src, size, stream);
```

### Explicit Data Transfers

To avoid inference overhead, use explicit copy functions:

```cpp
eIcicleError result = icicle_copy_to_host(host_dst, device_src, size);
eIcicleError result = icicle_copy_to_host_async(host_dst, device_src, size, stream);

eIcicleError result = icicle_copy_to_device(device_dst, host_src, size);
eIcicleError result = icicle_copy_to_device_async(device_dst, host_src, size, stream);
```

## Stream Management

### Creating and Destroying Streams

Streams are used to manage asynchronous operations:

```cpp
icicleStreamHandle stream;
eIcicleError result = icicle_create_stream(&stream);

eIcicleError result = icicle_destroy_stream(stream);
```

## Synchronization

### Synchronizing Streams and Devices

Ensure all previous operations on a stream or device are completed before proceeding:

```cpp
eIcicleError result = icicle_stream_synchronize(stream);

eIcicleError result = icicle_device_synchronize();
```

## Device Properties

### Querying Device Properties

Retrieve properties of the active device:

```cpp
DeviceProperties properties;
eIcicleError result = icicle_get_device_properties(properties);
```

### Checking Device Availability

Check if a device is available and retrieve a list of registered devices:

```cpp
icicle::Device dev;
eIcicleError result = icicle_is_device_avialable(dev);

char output[256];
eIcicleError result = icicle_get_registered_devices(output, sizeof(output));
```

## Compute APIs

The structure demonstrated in the following examples is common across all compute APIs in Icicle, including NTT, vector operations, ECNTT, and others. For detailed API usage and examples, please refer to the full API documentation.

### Multi-Scalar Multiplication (MSM) Example

Icicle provides high-performance compute APIs such as the Multi-Scalar Multiplication (MSM) for cryptographic operations. Here's a simple example of how to use the MSM API.

```cpp
#include <iostream>
#include "icicle/runtime.h"
#include "icicle/api/bn254.h"

using namespace bn254;

int main() {
  // Load backend and set device
    icicle_load_backend_from_env_or_default();

  // trying to choose CUDA if available, or fallback to CPU otherwise (default device)
  const bool is_cuda_device_available = (eIcicleError::SUCCESS == icicle_is_device_avialable("CUDA"));
  if (is_cuda_device_available) {
    Device device = {"CUDA", 0}; // GPU-0
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
  auto err = icicle_malloc((void**)&scalars_d, sizeof(scalar_t) * N);
  // Note: need to test err and make sure no errors occurred
  err = icicle_copy(scalars_d, scalars.get(), sizeof(scalar_t) * N);

  // MSM configuration
  MSMConfig config = default_msm_config();
  // tell icicle that the scalars are on device. Note that EC points and result are on host memory in this example.
  config.are_scalars_on_device = true;

  // Execute the MSM kernel (on the current device)
  eIcicleError result_code = bn254_msm(scalars_d, points.get(), msm_size, config, &result);
  
  // Free the device memory
  icicle_free(scalars_d);

  // Check for errors
  if (result_code == eIcicleError::SUCCESS) {
    std::cout << "MSM result: " << projective_t::to_affine(result) << std::endl;
  } else {
    std::cerr << "MSM computation failed with error: " << result_code << std::endl;
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

using namespace icicle;
using namespace bn254;

// define bn254Poly to be a polynomial over the scalar field of bn254
using bn254Poly = Polynomial<scalar_t>;

static bn254Poly randomize_polynomial(uint32_t size)
{
  auto coeff = std::make_unique<scalar_t[]>(size);
  for (int i = 0; i < size; i++)
    coeff[i] = scalar_t::rand_host();
  return bn254Poly::from_evaluations(coeff.get(), size);
}

int main() {
  // Load backend and set device
  icicle_load_backend_from_env_or_default();

  // trying to choose CUDA if available, or fallback to CPU otherwise (default device)
  const bool is_cuda_device_available = (eIcicleError::SUCCESS == icicle_is_device_avialable("CUDA"));
  if (is_cuda_device_available) {
    Device device = {"CUDA", 0}; // GPU-0
    ICICLE_CHECK(icicle_set_device(device)); // ICICLE_CHECK asserts that the API call returns eIcicleError::SUCCESS
  } // else we stay on CPU backend

  // randomize polynomials f(x),g(x) over the scalar field of bn254
  int poly_size = 1024;
  bn254Poly f = randomize_polynomial(poly_size);
  bn254Poly g = randomize_polynomial(poly_size);

  // Perform polynomial multiplication
  auto result = f * g; // Executes on the current device

  // Display result (or use result in further computations)
  std::cout << "Polynomial multiplication result: " << result << std::endl;

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
