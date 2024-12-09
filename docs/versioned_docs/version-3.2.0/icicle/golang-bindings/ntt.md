# NTT

## NTT Example

```go
package main

import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func init() {
  // Load backend using env path
	runtime.LoadBackendFromEnvOrDefault()
	// Set Cuda device to perform
	device := runtime.CreateDevice("CUDA", 0)
	runtime.SetDevice(&device)

	cfg := core.GetDefaultNTTInitDomainConfig()
	initDomain(18, cfg)
}

func initDomain(largestTestSize int, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := bn254.ScalarField{}
	limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

	rouIcicle.FromLimbs(limbs)
	e := ntt.InitDomain(rouIcicle, cfg)
	return e
}

func main() {
	// Obtain the default NTT configuration with a predefined coset generator.
	cfg := ntt.GetDefaultNttConfig()

	// Define the size of the input scalars.
	size := 1 << 18

	// Generate scalars for the NTT operation.
	scalars := bn254.GenerateScalars(size)

	// Set the direction of the NTT (forward or inverse).
	dir := core.KForward

	// Allocate memory for the results of the NTT operation.
	results := make(core.HostSlice[bn254.ScalarField], size)

	// Perform the NTT operation.
	err := ntt.Ntt(scalars, dir, &cfg, results)
	if err != runtime.Success {
		panic("NTT operation failed")
	}

	ntt.ReleaseDomain()
}
```

## NTT Method

```go
func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) runtime.EIcicleError
```

### Parameters

- **`scalars`**: A slice containing the input scalars for the transform. It can reside either in host memory or device memory.
- **`dir`**: The direction of the NTT operation (`KForward` or `KInverse`).
- **`cfg`**: A pointer to an `NTTConfig` object, containing configuration options for the NTT operation.
- **`results`**: A slice where the results of the NTT operation will be stored. This slice can be in host or device memory.

### Return Value

- **`EIcicleError`**: A `runtime.EIcicleError` value, which will be `runtime.Success` if the operation was successful, or an error if something went wrong.

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

### Initializing the NTT Domain

Before performing NTT operations, it's necessary to initialize the NTT domain; it only needs to be called once per GPU since the twiddles are cached.

```go
func InitDomain(primitiveRoot bn254.ScalarField, cfg core.NTTInitDomainConfig) runtime.EIcicleError
```

This function initializes the domain with a given primitive root, optionally using fast twiddle factors to optimize the computation.

### Releasing the domain

The `ReleaseDomain` function is responsible for releasing the resources associated with a specific domain in the CUDA device context.

```go
func ReleaseDomain() runtime.EIcicleError
```

### Return Value

- **`EIcicleError`**: A `runtime.EIcicleError` value, which will be `runtime.Success` if the operation was successful, or an error if something went wrong.
