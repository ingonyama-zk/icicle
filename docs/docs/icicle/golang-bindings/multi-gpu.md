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

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/msm"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func main() {
	// Load backend using env path
	runtime.LoadBackendFromEnvOrDefault()

	device := runtime.CreateDevice("CUDA", 0)
	err := runtime.SetDevice(&device)
	numDevices, _ := runtime.GetDeviceCount()
	fmt.Println("There are ", numDevices, " devices available")

	if err != runtime.Success {
		panic(err)
	}
	wg := sync.WaitGroup{}

	for i := 0; i < numDevices; i++ {
		internalDevice := runtime.Device{DeviceType: device.DeviceType, Id: int32(i)}
		wg.Add(1)
		runtime.RunOnDevice(&internalDevice, func(args ...any) {
			defer wg.Done()
			currentDevice, err := runtime.GetActiveDevice()
			if err != runtime.Success {
				panic("Failed to get current device")
			}

			fmt.Println("Running on ", currentDevice.GetDeviceType(), " ", currentDevice.Id, " device")

			cfg := msm.GetDefaultMSMConfig()
			cfg.IsAsync = true
			size := 1 << 10
			scalars := bn254.GenerateScalars(size)
			points := bn254.GenerateAffinePoints(size)

			stream, _ := runtime.CreateStream()
			var p bn254.Projective
			var out core.DeviceSlice
			_, err = out.MallocAsync(p.Size(), 1, stream)
			if err != runtime.Success {
				panic("Allocating bytes on device for Projective results failed")
			}
			cfg.StreamHandle = stream

			err = msm.Msm(scalars, points, &cfg, out)
			if err != runtime.Success {
				panic("Msm failed")
			}
			outHost := make(core.HostSlice[bn254.Projective], 1)
			outHost.CopyFromDeviceAsync(&out, stream)
			out.FreeAsync(stream)

			runtime.SynchronizeStream(stream)
			runtime.DestroyStream(stream)
			// Check with gnark-crypto
		})
	}
	wg.Wait()
}
```

This example demonstrates a basic pattern for distributing tasks across multiple GPUs. The `RunOnDevice` function ensures that each goroutine is executed on its designated GPU and a corresponding thread.

## Device Management API

To streamline device management we offer as part of `runtime` package methods for dealing with devices.

### `RunOnDevice`

Runs a given function on a specific GPU device, ensuring that all CUDA calls within the function are executed on the selected device.

In Go, most concurrency can be done via Goroutines. However, there is no guarantee that a goroutine stays on a specific host thread.

`RunOnDevice` was designed to solve this caveat and ensure that the goroutine will stay on a specific host thread.

`RunOnDevice` locks a goroutine into a specific host thread, sets a current GPU device, runs a provided function, and unlocks the goroutine from the host thread after the provided function finishes.

While the goroutine is locked to the host thread, the Go runtime will not assign other goroutines to that host thread.

**Parameters:**

- **`device *Device`**: A pointer to the `Device` instanse to be used to run code.
- **`funcToRun func(args ...any)`**: The function to be executed on the specified device.
- **`args ...any`**: Arguments to be passed to `funcToRun`.

**Behavior:**

- The function `funcToRun` is executed in a new goroutine that is locked to a specific OS thread to ensure that all CUDA calls within the function target the specified device.

:::note
Any goroutines launched within `funcToRun` are not automatically bound to the same GPU device. If necessary, `RunOnDevice` should be called again within such goroutines with the same `deviceId`.
:::

**Example:**

```go
device := runtime.CreateDevice("CUDA", 0)
RunOnDevice(&device, func(args ...any) {
	fmt.Println("This runs on GPU 0")
	// CUDA-related operations here will target GPU 0
}, nil)
```

### `SetDevice`

Sets the active device for the current host thread. All subsequent calls made from this thread will target the specified device.

:::warning
This function should not be used directly in conjunction with goroutines. If you want to run multi-gpu scenarios with goroutines you should use [RunOnDevice](#runondevice)
:::

**Parameters:**

- **`device *Device`**: A pointer to the `Device` instanse to be used to run code.

**Returns:**

- **`EIcicleError`**: A `runtime.EIcicleError` value, which will be `runtime.Success` if the operation was successful, or an error if something went wrong.

### `GetDeviceCount`

Retrieves the number of devices available on the host.

**Returns:**

- **`(int, EIcicleError)`**: The number of devices and an error code indicating the success or failure of the operation.

### `GetActiveDevice`

Gets the device of the currently active device for the calling host thread.

**Returns:**

- **`(*Device, EIcicleError)`**: The device pointer and an error code indicating the success or failure of the operation.


This documentation should provide a clear understanding of how to effectively manage multiple GPUs in Go applications using CUDA and other backends, with a particular emphasis on the `RunOnDevice` function for executing tasks on specific GPUs.
