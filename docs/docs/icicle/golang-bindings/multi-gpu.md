# Multi GPU APIs

To learn more about the theory of Multi GPU programming refer to [this part](../multi-gpu.md) of documentation.

Here we will cover the core multi GPU apis and an [example](#a-multi-gpu-example)

## A Multi GPU example

In this example we will display how you can

1. Fetch the number of devices installed on a machine
2. For every GPU launch a thread and set an active device per thread.
3. Execute a MSM on each GPU

```go
package main

import (
	"fmt"
	"sync"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254"
)

func main() {
	numDevices, _ := cr.GetDeviceCount()
	fmt.Println("There are ", numDevices, " devices available")
	wg := sync.WaitGroup{}

	for i := 0; i < numDevices; i++ {
		wg.Add(1)
		// RunOnDevice makes sure each MSM runs on a single thread
		cr.RunOnDevice(i, func(args ...any) {
			defer wg.Done()
			cfg := bn254.GetDefaultMSMConfig()
			cfg.IsAsync = true
			for _, power := range []int{10, 18} {
				size := 1 << power // 2^pwr

				// generate random scalars
				scalars := bn254.GenerateScalars(size)
				points := bn254.GenerateAffinePoints(size)

				// create a stream and allocate result pointer
				stream, _ := cr.CreateStream()
				var p bn254.Projective
				var out core.DeviceSlice
				out.MallocAsync(p.Size(), p.Size(), stream)
				// assign stream to device context
				cfg.Ctx.Stream = &stream

				// execute MSM
				bn254.Msm(scalars, points, &cfg, out)
				// read result from device
				outHost := make(core.HostSlice[bn254.Projective], 1)
				outHost.CopyFromDeviceAsync(&out, stream)
				out.FreeAsync(stream)

				// sync the stream
				cr.SynchronizeStream(&stream)
			}
		})
	}
	wg.Wait()
}
```

This example demonstrates a basic pattern for distributing tasks across multiple GPUs. The `RunOnDevice` function ensures that each goroutine is executed on its designated GPU and a corresponding thread.

## Device Management API

To streamline device management we offer as part of `cuda_runtime` package methods for dealing with devices.

### `RunOnDevice`

Runs a given function on a specific GPU device, ensuring that all CUDA calls within the function are executed on the selected device.

In Go, most concurrency can be done via Goroutines. However, there is no guarantee that a goroutine stays on a specific host thread.

`RunOnDevice` was designed to solve this caveat and ensure that the goroutine will stay on a specific host thread.

`RunOnDevice` locks a goroutine into a specific host thread, sets a current GPU device, runs a provided function, and unlocks the goroutine from the host thread after the provided function finishes.

While the goroutine is locked to the host thread, the Go runtime will not assign other goroutines to that host thread.

**Parameters:**

- **`deviceId int`**: The ID of the device on which to run the provided function. Device IDs start from 0.
- **`funcToRun func(args ...any)`**: The function to be executed on the specified device.
- **`args ...any`**: Arguments to be passed to `funcToRun`.

**Behavior:**

- The function `funcToRun` is executed in a new goroutine that is locked to a specific OS thread to ensure that all CUDA calls within the function target the specified device.

:::note
Any goroutines launched within `funcToRun` are not automatically bound to the same GPU device. If necessary, `RunOnDevice` should be called again within such goroutines with the same `deviceId`.
:::

**Example:**

```go
RunOnDevice(0, func(args ...any) {
	fmt.Println("This runs on GPU 0")
	// CUDA-related operations here will target GPU 0
}, nil)
```

### `SetDevice`

Sets the active device for the current host thread. All subsequent CUDA calls made from this thread will target the specified device.

:::warning
This function should not be used directly in conjunction with goroutines. If you want to run multi-gpu scenarios with goroutines you should use [RunOnDevice](#runondevice)
:::

**Parameters:**

- **`device int`**: The ID of the device to set as the current device.

**Returns:**

- **`CudaError`**: Error code indicating the success or failure of the operation.

### `GetDeviceCount`

Retrieves the number of CUDA-capable devices available on the host.

**Returns:**

- **`(int, CudaError)`**: The number of devices and an error code indicating the success or failure of the operation.

### `GetDevice`

Gets the ID of the currently active device for the calling host thread.

**Returns:**

- **`(int, CudaError)`**: The ID of the current device and an error code indicating the success or failure of the operation.

### `GetDeviceFromPointer`

Retrieves the device associated with a given pointer.

**Parameters:**

- **`ptr unsafe.Pointer`**: Pointer to query.

**Returns:**

- **`int`**: The device ID associated with the memory pointed to by `ptr`.

This documentation should provide a clear understanding of how to effectively manage multiple GPUs in Go applications using CUDA, with a particular emphasis on the `RunOnDevice` function for executing tasks on specific GPUs.
