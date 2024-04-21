# Vector Operations

## Overview

The VecOps API provides efficient vector operations such as addition, subtraction, and multiplication.

## Example

### Vector addition

```go
package main

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254"
)

func main() {
	testSize := 1 << 12
	a := bn254.GenerateScalars(testSize)
	b := bn254.GenerateScalars(testSize)
	out := make(core.HostSlice[bn254.ScalarField], testSize)
	cfg := core.DefaultVecOpsConfig()

	// Perform vector multiplication
	err := bn254.VecOp(a, b, out, cfg, core.Add)
	if err != cr.CudaSuccess {
		panic("Vector addition failed")
	}
}
```

### Vector Subtraction

```go
package main

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254"
)

func main() {
	testSize := 1 << 12
	a := bn254.GenerateScalars(testSize)
	b := bn254.GenerateScalars(testSize)
	out := make(core.HostSlice[bn254.ScalarField], testSize)
	cfg := core.DefaultVecOpsConfig()

	// Perform vector multiplication
	err := bn254.VecOp(a, b, out, cfg, core.Sub)
	if err != cr.CudaSuccess {
		panic("Vector subtraction failed")
	}
}
```

### Vector Multiplication

```go
package main

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254"
)

func main() {
	testSize := 1 << 12
	a := bn254.GenerateScalars(testSize)
	b := bn254.GenerateScalars(testSize)
	out := make(core.HostSlice[bn254.ScalarField], testSize)
	cfg := core.DefaultVecOpsConfig()

	// Perform vector multiplication
	err := bn254.VecOp(a, b, out, cfg, core.Mul)
	if err != cr.CudaSuccess {
		panic("Vector multiplication failed")
	}
}
```

## VecOps Method

```go
func VecOp(a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret cr.CudaError)
```

### Parameters

- **a**: The first input vector.
- **b**: The second input vector.
- **out**: The output vector where the result of the operation will be stored.
- **config**: A `VecOpsConfig` object containing various configuration options for the vector operations.
- **op**: The operation to perform, specified as one of the constants (`Sub`, `Add`, `Mul`) from the `VecOps` type.

### Return Value

- **CudaError**: Returns a CUDA error code indicating the success or failure of the vector operation.

## VecOpsConfig

The `VecOpsConfig` structure holds configuration parameters for the vector operations, allowing customization of its behavior.

```go
type VecOpsConfig struct {
    Ctx cr.DeviceContext
    isAOnDevice bool
    isBOnDevice bool
    isResultOnDevice bool
    IsAsync bool
}
```

### Fields

- **Ctx**: Device context containing details like device ID and stream ID.
- **isAOnDevice**: Indicates if vector `a` is located on the device.
- **isBOnDevice**: Indicates if vector `b` is located on the device.
- **isResultOnDevice**: Specifies where the result vector should be stored (device or host memory).
- **IsAsync**: Controls whether the vector operation runs asynchronously.

### Default Configuration

Use `DefaultVecOpsConfig` to obtain a default configuration, customizable as needed.

```go
func DefaultVecOpsConfig() VecOpsConfig
```
