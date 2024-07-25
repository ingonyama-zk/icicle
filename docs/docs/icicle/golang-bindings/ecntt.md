# ECNTT

## ECNTT Method

The `ECNtt[T any]()` function performs the Elliptic Curve Number Theoretic Transform (EC-NTT) on the input points slice, using the provided dir (direction), cfg (configuration), and stores the results in the results slice.

```go
func ECNtt[T any](points core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) core.IcicleError
```

### Parameters

- **`points`**: A slice of elliptic curve points (in projective coordinates) that will be transformed. The slice can be stored on the host or the device, as indicated by the `core.HostOrDeviceSlice` type.
- **`dir`**: The direction of the EC-NTT transform, either `core.KForward` or `core.KInverse`.
- **`cfg`**: A pointer to an `NTTConfig` object, containing configuration options for the NTT operation.
- **`results`**: A slice that will store the transformed elliptic curve points (in projective coordinates). The slice can be stored on the host or the device, as indicated by the `core.HostOrDeviceSlice` type.

### Return Value

- **`CudaError`**: A `core.IcicleError` value, which will be `core.IcicleErrorCode(0)` if the EC-NTT operation was successful, or an error if something went wrong.

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

- **`Ctx`**: Device context containing details like device ID and stream ID.
- **`CosetGen`**: Coset generator used for coset (i)NTTs, defaulting to no coset being used.
- **`BatchSize`**: The number of NTTs to compute in one operation, defaulting to 1.
- **`ColumnsBatch`**: If true the function will compute the NTTs over the columns of the input matrix and not over the rows. Defaults to `false`.
- **`Ordering`**: Ordering of inputs and outputs (`KNN`, `KNR`, `KRN`, `KRR`), affecting how data is arranged.
- **`areInputsOnDevice`**: Indicates if input scalars are located on the device.
- **`areOutputsOnDevice`**: Indicates if results are stored on the device.
- **`IsAsync`**: Controls whether the NTT operation runs asynchronously.
- **`NttAlgorithm`**: Explicitly select the NTT algorithm. ECNTT supports running on `Radix2` algorithm.

### Default Configuration

Use `GetDefaultNTTConfig` to obtain a default configuration, customizable as needed.

```go
func GetDefaultNTTConfig[T any](cosetGen T) NTTConfig[T]
```

## ECNTT Example

```go
package main

import (
    "github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
    cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

func Main() {
    // Obtain the default NTT configuration with a predefined coset generator.
    cfg := GetDefaultNttConfig()
    
    // Define the size of the input scalars.
    size := 1 << 18

    // Generate Points for the ECNTT operation.
    points := GenerateProjectivePoints(size)
    
    // Set the direction of the NTT (forward or inverse).
    dir := core.KForward

    // Allocate memory for the results of the NTT operation.
    results := make(core.HostSlice[Projective], size)

    // Perform the NTT operation.
    err := ECNtt(points, dir, &cfg, results)
    if err != cr.CudaSuccess {
        panic("ECNTT operation failed")
    }
}
```
