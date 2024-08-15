# Icicle C++ Usage Guide

## Overview

This guide will cover the usage of Icicle's C++ API, including device management, memory operations, data transfer, and synchronization.

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

## Error Handling

### Checking for Errors

Icicle APIs return an `eIcicleError` enumeration value. Always check the returned value to ensure that operations were successful.

```cpp
if (result != eIcicleError::SUCCESS) {
    // Handle error
}
```

This guide provides an overview of the essential APIs available in Icicle for C++. The provided examples should help you get started with integrating Icicle into your high-performance computing projects.
