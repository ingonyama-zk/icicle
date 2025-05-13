# MSM

## MSM Example

```go
package main

import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/msm"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func main() {
	// Load backend using env path
	runtime.LoadBackendFromEnvOrDefault()
	// Set Cuda device to perform
	device := runtime.CreateDevice("CUDA", 0)
	runtime.SetDevice(&device)

	// Obtain the default MSM configuration.
	cfg := core.GetDefaultMSMConfig()

	// Define the size of the problem, here 2^18.
	size := 1 << 18

	// Generate scalars and points for the MSM operation.
	scalars := bn254.GenerateScalars(size)
	points := bn254.GenerateAffinePoints(size)

	// Create a CUDA stream for asynchronous operations.
	stream, _ := runtime.CreateStream()
	var p bn254.Projective

	// Allocate memory on the device for the result of the MSM operation.
	var out core.DeviceSlice
	_, e := out.MallocAsync(p.Size(), 1, stream)

	if e != runtime.Success {
		panic(e)
	}

	// Set the CUDA stream in the MSM configuration.
	cfg.StreamHandle = stream
	cfg.IsAsync = true

	// Perform the MSM operation.
	e = msm.Msm(scalars, points, &cfg, out)

	if e != runtime.Success {
		panic(e)
	}

	// Allocate host memory for the results and copy the results from the device.
	outHost := make(core.HostSlice[bn254.Projective], 1)
	runtime.SynchronizeStream(stream)
	runtime.DestroyStream(stream)
	outHost.CopyFromDevice(&out)

	// Free the device memory allocated for the results.
	out.Free()
}
```

## MSM Method

```go
func Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) runtime.EIcicleError
```

### Parameters

- **`scalars`**: A slice containing the scalars for multiplication. It can reside either in host memory or device memory.
- **`points`**: A slice containing the points to be multiplied with scalars. Like scalars, these can also be in host or device memory.
- **`cfg`**: A pointer to an `MSMConfig` object, which contains various configuration options for the MSM operation.
- **`results`**: A slice where the results of the MSM operation will be stored. This slice can be in host or device memory.

### Return Value

- **`EIcicleError`**: A `runtime.EIcicleError` value, which will be `runtime.Success` if the operation was successful, or an error if something went wrong.

## MSMConfig

The `MSMConfig` structure holds configuration parameters for the MSM operation, allowing customization of its behavior to optimize performance based on the specifics of the operation or the underlying hardware.

```go
type MSMConfig struct {
	StreamHandle             runtime.Stream
	PrecomputeFactor         int32
	C                        int32
	Bitsize                  int32
	BatchSize                int32
	ArePointsSharedInBatch   bool
	areScalarsOnDevice       bool
	AreScalarsMontgomeryForm bool
	areBasesOnDevice         bool
	AreBasesMontgomeryForm   bool
	areResultsOnDevice       bool
	IsAsync                  bool
	Ext                      config_extension.ConfigExtensionHandler
}
```

### Fields

- **`StreamHandle`**: Specifies the stream (queue) to use for async execution.
- **`PrecomputeFactor`**: Controls the number of extra points to pre-compute.
- **`C`**: Window bitsize, a key parameter in the "bucket method" for MSM.
- **`Bitsize`**: Number of bits of the largest scalar.
- **`BatchSize`**: Number of results to compute in one batch.
- **`ArePointsSharedInBatch`**: Bases are shared for batch. Set to true if all MSMs use the same bases. Otherwise, the number of bases and number of scalars are expected to be equal.
- **`areScalarsOnDevice`**: Indicates if scalars are located on the device.
- **`AreScalarsMontgomeryForm`**: True if scalars are in Montgomery form.
- **`areBasesOnDevice`**: Indicates if bases are located on the device.
- **`AreBasesMontgomeryForm`**: True if point coordinates are in Montgomery form.
- **`areResultsOnDevice`**: Indicates if results are stored on the device.
- **`IsAsync`**: If true, runs MSM asynchronously.
- **`Ext`**: Extended configuration for backend.

### Default Configuration

Use `GetDefaultMSMConfig` to obtain a default configuration, which can then be customized as needed.

```go
func GetDefaultMSMConfig() MSMConfig
```

## Batched msm

For batch msm, simply allocate the results array with size corresponding to batch size and set the `ArePointsSharedInBatch` flag in config struct.

```go
...

// Obtain the default MSM configuration.
cfg := GetDefaultMSMConfig()

cfg.Ctx.IsBigTriangle = true

...
```

## How do I toggle between MSM modes?

Toggling between MSM modes occurs automatically based on the number of results you are expecting from the `MSM` function.

The number of results is interpreted from the size of `var out core.DeviceSlice`. Thus its important when allocating memory for `var out core.DeviceSlice` to make sure that you are allocating `<number of results> X <size of a single point>`.

```go
... 

batchSize := 3
var p G2Projective
var out core.DeviceSlice
out.Malloc(p.Size(), batchSize)

...
```

## Parameters for optimal performance

Please refer to the [primitive description](../primitives/msm#choosing-optimal-parameters)

## Support for G2 group

To activate G2 support first you must make sure you are building the static libraries with G2 feature enabled as described in the [Golang building instructions](../golang-bindings.md#using-icicle-golang-bindings-in-your-project).

Now you may import `g2` package of the specified curve.

```go
import (
    "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/g2"
)
```

This package include `G2Projective` and `G2Affine` points as well as a `G2Msm` method.

```go
package main

import (
	"log"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/msm"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func main() {
	cfg := core.GetDefaultMSMConfig()
	points := bn254.GenerateAffinePoints(1024)
	var precomputeFactor int32 = 8
	var precomputeOut core.DeviceSlice
	precomputeOut.Malloc(points[0].Size(), points.Len()*int(precomputeFactor))

	err := msm.PrecomputeBases(points, &cfg, precomputeOut)
	if err != runtime.Success {
		log.Fatalf("PrecomputeBases failed: %v", err)
	}
}
```

`G2Msm` works the same way as normal MSM, the difference is that it uses G2 Points.
