# ECNTT

## ECNTT Method

The `ECNtt[T any]()` function performs the Elliptic Curve Number Theoretic Transform (EC-NTT) on the input points slice, using the provided dir (direction), cfg (configuration), and stores the results in the results slice.

```go
func ECNtt[T any](points core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) runtime.EIcicleError
```

### Parameters

- **`points`**: A slice of elliptic curve points (in projective coordinates) that will be transformed. The slice can be stored on the host or the device, as indicated by the `core.HostOrDeviceSlice` type.
- **`dir`**: The direction of the EC-NTT transform, either `core.KForward` or `core.KInverse`.
- **`cfg`**: A pointer to an `NTTConfig` object, containing configuration options for the NTT operation.
- **`results`**: A slice that will store the transformed elliptic curve points (in projective coordinates). The slice can be stored on the host or the device, as indicated by the `core.HostOrDeviceSlice` type.

### Return Value

- **`EIcicleError`**: A `runtime.EIcicleError` value, which will be `runtime.Success` if the EC-NTT operation was successful, or an error if something went wrong.

## NTT Configuration (NTTConfig)

The `NTTConfig` structure holds configuration parameters for the NTT operation, allowing customization of its behavior to optimize performance based on the specifics of your protocol.

```go
type NTTConfig[T any] struct {
	StreamHandle       runtime.Stream
	CosetGen           T
	BatchSize          int32
	ColumnsBatch       bool
	Ordering           Ordering
	areInputsOnDevice  bool
	areOutputsOnDevice bool
	IsAsync            bool
	Ext                config_extension.ConfigExtensionHandler
}
```

### Fields

- **`StreamHandle`**: Specifies the stream (queue) to use for async execution.
- **`CosetGen`**: Coset generator. Used to perform coset (i)NTTs.
- **`BatchSize`**: The number of NTTs to compute in one operation, defaulting to 1.
- **`ColumnsBatch`**: If true the function will compute the NTTs over the columns of the input matrix and not over the rows.
- **`Ordering`**: Ordering of inputs and outputs (`KNN`, `KNR`, `KRN`, `KRR`), affecting how data is arranged.
- **`areInputsOnDevice`**: Indicates if input scalars are located on the device.
- **`areOutputsOnDevice`**: Indicates if results are stored on the device.
- **`IsAsync`**: Controls whether the NTT operation runs asynchronously.
- **`Ext`**: Extended configuration for backend.

### Default Configuration

Use `GetDefaultNTTConfig` to obtain a default configuration, customizable as needed.

```go
func GetDefaultNTTConfig[T any](cosetGen T) NTTConfig[T]
```

## ECNTT Example

```go
package main

import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ecntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func Main() {
	// Load backend using env path
	runtime.LoadBackendFromEnvOrDefault()
	// Set Cuda device to perform
	device := runtime.CreateDevice("CUDA", 0)
	runtime.SetDevice(&device)
	// Obtain the default NTT configuration with a predefined coset generator.
	cfg := ntt.GetDefaultNttConfig()

	// Define the size of the input scalars.
	size := 1 << 18

	// Generate Points for the ECNTT operation.
	points := bn254.GenerateProjectivePoints(size)

	// Set the direction of the NTT (forward or inverse).
	dir := core.KForward

	// Allocate memory for the results of the NTT operation.
	results := make(core.HostSlice[bn254.Projective], size)

	// Perform the NTT operation.
	err := ecntt.ECNtt(points, dir, &cfg, results)
	if err != runtime.Success {
		panic("ECNTT operation failed")
	}
}
```
