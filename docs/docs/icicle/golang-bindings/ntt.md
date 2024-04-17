# NTT

### Supported curves

`bls12-377`, `bls12-381`, `bn254`, `bw6-761`

## NTT Example

```go
package main

import (
    "github.com/ingonyama-zk/icicle/wrappers/golang/core"
    cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func Main() {
    // Obtain the default NTT configuration with a predefined coset generator.
    cfg := GetDefaultNttConfig()
    
    // Define the size of the input scalars.
    size := 1 << 18

    // Generate scalars for the NTT operation.
    scalars := GenerateScalars(size)

    // Set the direction of the NTT (forward or inverse).
    dir := core.KForward

    // Allocate memory for the results of the NTT operation.
    results := make(core.HostSlice[ScalarField], size)

    // Perform the NTT operation.
    err := Ntt(scalars, dir, &cfg, results)
    if err != cr.CudaSuccess {
        panic("NTT operation failed")
    }
}
```

## NTT Method

```go
func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) core.IcicleError
```

### Parameters

- **scalars**: A slice containing the input scalars for the transform. It can reside either in host memory or device memory.
- **dir**: The direction of the NTT operation (`KForward` or `KInverse`).
- **cfg**: A pointer to an `NTTConfig` object, containing configuration options for the NTT operation.
- **results**: A slice where the results of the NTT operation will be stored. This slice can be in host or device memory.

### Return Value

- **CudaError**: Returns a CUDA error code indicating the success or failure of the NTT operation.

## NTT Configuration (NTTConfig)

The `NTTConfig` structure holds configuration parameters for the NTT operation, allowing customization of its behavior to optimize performance based on the specifics of your protocol.

```go
type NTTConfig[T any] struct {
    Ctx cr.DeviceContext
    CosetGen T
    BatchSize int32
    ColumnsBatch bool
    Ordering Ordering
    areInputsOnDevice  bool
    areOutputsOnDevice bool
    IsAsync bool
    NttAlgorithm NttAlgorithm
}
```

### Fields

- **Ctx**: Device context containing details like device ID and stream ID.
- **CosetGen**: Coset generator used for coset (i)NTTs, defaulting to no coset being used.
- **BatchSize**: The number of NTTs to compute in one operation, defaulting to 1.
- **ColumnsBatch**: If true the function will compute the NTTs over the columns of the input matrix and not over the rows. Defaults to `false`.
- **Ordering**: Ordering of inputs and outputs (`KNN`, `KNR`, `KRN`, `KRR`, `KMN`, `KNM`), affecting how data is arranged.
- **areInputsOnDevice**: Indicates if input scalars are located on the device.
- **areOutputsOnDevice**: Indicates if results are stored on the device.
- **IsAsync**: Controls whether the NTT operation runs asynchronously.
- **NttAlgorithm**: Explicitly select the NTT algorithm. Default value: Auto (the implementation selects radix-2 or mixed-radix algorithm based on heuristics).

### Default Configuration

Use `GetDefaultNTTConfig` to obtain a default configuration, customizable as needed.

```go
func GetDefaultNTTConfig[T any](cosetGen T) NTTConfig[T]
```

### Initializing the NTT Domain

Before performing NTT operations, it's necessary to initialize the NTT domain; it only needs to be called once per GPU since the twiddles are cached.

```go
func InitDomain(primitiveRoot ScalarField, ctx cr.DeviceContext, fastTwiddles bool) core.IcicleError
```

This function initializes the domain with a given primitive root, optionally using fast twiddle factors to optimize the computation.

### Releasing the domain

The `ReleaseDomain` function is responsible for releasing the resources associated with a specific domain in the CUDA device context.

```go
func ReleaseDomain(ctx cr.DeviceContext) core.IcicleError
```

### Parameters

- **`ctx`**: a reference to the `DeviceContext` object, which represents the CUDA device context.

### Return Value

The function returns a `core.IcicleError`, which represents the result of the operation. If the operation is successful, the function returns `core.IcicleErrorCode(0)`.

### Example

```go
import (
    "github.com/icicle-crypto/icicle-core/cr"
    "github.com/icicle-crypto/icicle-core/core"
)

func example() {
    cfg := GetDefaultNttConfig()
	err := ReleaseDomain(cfg.Ctx)
    if err != nil {
        // Handle the error
    }
}
```
