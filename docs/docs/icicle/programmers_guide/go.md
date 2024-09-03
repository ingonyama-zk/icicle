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

## Compute APIs

### Multi-Scalar Multiplication (MSM) Example

Icicle provides high-performance compute APIs such as the Multi-Scalar Multiplication (MSM) for cryptographic operations. Here's a simple example of how to use the MSM API.

```go
package main

import (
	"fmt"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	bn254Msm "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/msm"
)

func main() {

	// Load installed backends
	runtime.LoadBackendFromEnvOrDefault()

	// trying to choose CUDA if available, or fallback to CPU otherwise (default device)
	deviceCuda := runtime.CreateDevice("CUDA", 0) // GPU-0
	if runtime.IsDeviceAvailable(&deviceCuda) {
		runtime.SetDevice(&deviceCuda)
	} // else we stay on CPU backend

	// Setup inputs
	const size = 1 << 18

	// Generate random inputs
	scalars := bn254.GenerateScalars(size)
	points := bn254.GenerateAffinePoints(size)

	// (optional) copy scalars to device memory explicitly
	var scalarsDevice core.DeviceSlice
	scalars.CopyToDevice(&scalarsDevice, true)

	// MSM configuration
	cfgBn254 := core.GetDefaultMSMConfig()

	// allocate memory for the result
	result := make(core.HostSlice[bn254.Projective], 1)

	// execute bn254 MSM on device
	err := bn254Msm.Msm(scalarsDevice, points, &cfgBn254, result)

	// Check for errors
	if err != runtime.Success {
		errorString := fmt.Sprint(
			"bn254 Msm failed: ", err)
		panic(errorString)
	}

	// free explicitly allocated device memory
	scalarsDevice.Free()
}
```

### Polynomial Operations Example

Here's another example demonstrating polynomial operations using Icicle:

```go
package main

import (
	"fmt"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/polynomial"
)

func initBabybearDomain() runtime.EIcicleError {
	cfgInitDomain := core.GetDefaultNTTInitDomainConfig()
	rouIcicle := babybear.ScalarField{}
	rouIcicle.FromUint32(1461624142)
	return ntt.InitDomain(rouIcicle, cfgInitDomain)
}

func init() {
	// Load installed backends
	runtime.LoadBackendFromEnvOrDefault()

	// trying to choose CUDA if available, or fallback to CPU otherwise (default device)
	deviceCuda := runtime.CreateDevice("CUDA", 0) // GPU-0
	if runtime.IsDeviceAvailable(&deviceCuda) {
		runtime.SetDevice(&deviceCuda)
	} // else we stay on CPU backend

	// build domain for ntt is required for some polynomial ops that rely on ntt
	err := initBabybearDomain()
	if err != runtime.Success {
		errorString := fmt.Sprint(
			"Babybear Domain initialization failed: ", err)
		panic(errorString)
	}
}

func main() {

	// Setup inputs
	const polySize = 1 << 10

	// randomize two polynomials over babybear field
	var fBabybear polynomial.DensePolynomial
	defer fBabybear.Delete()
	var gBabybear polynomial.DensePolynomial
	defer gBabybear.Delete()
	fBabybear.CreateFromCoeffecitients(babybear.GenerateScalars(polySize))
	gBabybear.CreateFromCoeffecitients(babybear.GenerateScalars(polySize / 2))

	// Perform polynomial multiplication
	rBabybear := fBabybear.Multiply(&gBabybear) // Executes on the current device
	defer rBabybear.Delete()
	rDegree := rBabybear.Degree()

	fmt.Println("f Degree: ", fBabybear.Degree())
	fmt.Println("g Degree: ", gBabybear.Degree())
	fmt.Println("r Degree: ", rDegree)
}
```

In this example, the polynomial multiplication is used to perform polynomial multiplication on CUDA or CPU, showcasing the flexibility and power of Icicle's compute APIs.

## Error Handling

### Checking for Errors

Icicle APIs return an `EIcicleError` enumeration value. Always check the returned value to ensure that operations were successful.

```go
if result != runtime.SUCCESS {
    // Handle error
}
```

This guide provides an overview of the essential APIs available in Icicle for C++. The provided examples should help you get started with integrating Icicle into your high-performance computing projects.
