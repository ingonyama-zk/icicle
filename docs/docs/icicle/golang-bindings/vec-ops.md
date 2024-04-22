# Vector Operations

## Overview
Icicle is exposing a number of vector operations which a user can control:
* The VecOps API provides efficient vector operations such as addition, subtraction, and multiplication.
* MatrixTranspose API allows a user to perform a transpose on a vector representation of a matrix


## VecOps API Documentation
### Example

#### Vector addition

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

#### Vector Subtraction

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

#### Vector Multiplication

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

### VecOps Method

```go
func VecOp(a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret cr.CudaError)
```

#### Parameters

- **`a`**: The first input vector.
- **`b`**: The second input vector.
- **`out`**: The output vector where the result of the operation will be stored.
- **`config`**: A `VecOpsConfig` object containing various configuration options for the vector operations.
- **`op`**: The operation to perform, specified as one of the constants (`Sub`, `Add`, `Mul`) from the `VecOps` type.

#### Return Value

- **`CudaError`**: Returns a CUDA error code indicating the success or failure of the vector operation.

### VecOpsConfig

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

#### Fields

- **Ctx**: Device context containing details like device ID and stream ID.
- **isAOnDevice**: Indicates if vector `a` is located on the device.
- **isBOnDevice**: Indicates if vector `b` is located on the device.
- **isResultOnDevice**: Specifies where the result vector should be stored (device or host memory).
- **IsAsync**: Controls whether the vector operation runs asynchronously.

#### Default Configuration

Use `DefaultVecOpsConfig` to obtain a default configuration, customizable as needed.

```go
func DefaultVecOpsConfig() VecOpsConfig
```

## MatrixTranspose API Documentation

This section describes the functionality of the `TransposeMatrix` function used for matrix transposition.

The function takes a matrix represented as a 1D slice and transposes it, storing the result in another 1D slice.

### Function

```go
func TransposeMatrix(in, out core.HostOrDeviceSlice, columnSize, rowSize int, ctx cr.DeviceContext, onDevice, isAsync bool) (ret core.IcicleError)
```

## Parameters

- **`in`**: The input matrix is a `core.HostOrDeviceSlice`, stored as a 1D slice.
- **`out`**: The output matrix is a `core.HostOrDeviceSlice`, which will be the transpose of the input matrix, stored as a 1D slice.
- **`columnSize`**: The number of columns in the input matrix.
- **`rowSize`**: The number of rows in the input matrix.
- **`ctx`**: The device context `cr.DeviceContext` to be used for the matrix transpose operation.
- **`onDevice`**: Indicates whether the input and output slices are stored on the device (GPU) or the host (CPU).
- **`isAsync`**: Indicates whether the matrix transpose operation should be executed asynchronously.

## Return Value

The function returns a `core.IcicleError` value, which represents the result of the matrix transpose operation. If the operation is successful, the returned value will be `0`.

## Example Usage

```go
var input = make(core.HostSlice[ScalarField], 20)
var output = make(core.HostSlice[ScalarField], 20)

// Populate the input matrix
// ...

// Get device context
ctx, _ := cr.GetDefaultDeviceContext()

// Transpose the matrix
err := TransposeMatrix(input, output, 5, 4, ctx, false, false)
if err.IcicleErrorCode != core.IcicleErrorCode(0) {
    // Handle the error
}

// Use the transposed matrix
// ...
```

In this example, the `TransposeMatrix` function is used to transpose a 5x4 matrix stored in a 1D slice. The input and output slices are stored on the host (CPU), and the operation is executed synchronously.