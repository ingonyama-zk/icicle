# MSM


### Supported curves

`bls12-377`, `bls12-381`, `bn254`, `bw6-761`, `grumpkin`

## MSM Example

```go
package main

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254"
)

func main() {
	// Obtain the default MSM configuration.
	cfg := bn254.GetDefaultMSMConfig()

	// Define the size of the problem, here 2^18.
	size := 1 << 18

	// Generate scalars and points for the MSM operation.
	scalars := bn254.GenerateScalars(size)
	points := bn254.GenerateAffinePoints(size)

	// Create a CUDA stream for asynchronous operations.
	stream, _ := cr.CreateStream()
	var p bn254.Projective

	// Allocate memory on the device for the result of the MSM operation.
	var out core.DeviceSlice
	_, e := out.MallocAsync(p.Size(), p.Size(), stream)

	if e != cr.CudaSuccess {
		panic(e)
	}

	// Set the CUDA stream in the MSM configuration.
	cfg.Ctx.Stream = &stream
	cfg.IsAsync = true

	// Perform the MSM operation.
	e = bn254.Msm(scalars, points, &cfg, out)

	if e != cr.CudaSuccess {
		panic(e)
	}

	// Allocate host memory for the results and copy the results from the device.
	outHost := make(core.HostSlice[bn254.Projective], 1)
	cr.SynchronizeStream(&stream)
	outHost.CopyFromDevice(&out)

	// Free the device memory allocated for the results.
	out.Free()
}

```

## MSM Method

```go
func Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError
```

### Parameters

- **`scalars`**: A slice containing the scalars for multiplication. It can reside either in host memory or device memory.
- **`points`**: A slice containing the points to be multiplied with scalars. Like scalars, these can also be in host or device memory.
- **`cfg`**: A pointer to an `MSMConfig` object, which contains various configuration options for the MSM operation.
- **`results`**: A slice where the results of the MSM operation will be stored. This slice can be in host or device memory.

### Return Value

- **`CudaError`**: Returns a CUDA error code indicating the success or failure of the MSM operation.

## MSMConfig

The `MSMConfig` structure holds configuration parameters for the MSM operation, allowing customization of its behavior to optimize performance based on the specifics of the operation or the underlying hardware.

```go
type MSMConfig struct {
    Ctx cr.DeviceContext
    PrecomputeFactor int32
    C int32
    Bitsize int32
    LargeBucketFactor int32
    batchSize int32
    areScalarsOnDevice bool
    AreScalarsMontgomeryForm bool
    arePointsOnDevice bool
    ArePointsMontgomeryForm bool
    areResultsOnDevice bool
    IsBigTriangle bool
    IsAsync bool
}
```

### Fields

- **`Ctx`**: Device context containing details like device id and stream.
- **`PrecomputeFactor`**: Controls the number of extra points to pre-compute.
- **`C`**: Window bitsize, a key parameter in the "bucket method" for MSM.
- **`Bitsize`**: Number of bits of the largest scalar.
- **`LargeBucketFactor`**: Sensitivity to frequently occurring buckets.
- **`batchSize`**: Number of results to compute in one batch.
- **`areScalarsOnDevice`**: Indicates if scalars are located on the device.
- **`AreScalarsMontgomeryForm`**: True if scalars are in Montgomery form.
- **`arePointsOnDevice`**: Indicates if points are located on the device.
- **`ArePointsMontgomeryForm`**: True if point coordinates are in Montgomery form.
- **`areResultsOnDevice`**: Indicates if results are stored on the device.
- **`IsBigTriangle`**: If `true` MSM will run in Large triangle accumulation if `false` Bucket accumulation will be chosen. Default value: false.
- **`IsAsync`**: If true, runs MSM asynchronously.

### Default Configuration

Use `GetDefaultMSMConfig` to obtain a default configuration, which can then be customized as needed.

```go
func GetDefaultMSMConfig() MSMConfig
```


## How do I toggle between the supported algorithms?

When creating your MSM Config you may state which algorithm you wish to use. `cfg.Ctx.IsBigTriangle = true` will activate Large triangle accumulation and `cfg.Ctx.IsBigTriangle = false` will activate Bucket accumulation.

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
out.Malloc(batchSize*p.Size(), p.Size())

...
```

## Support for G2 group

To activate G2 support first you must make sure you are building the static libraries with G2 feature enabled.

```bash
./build.sh bn254 -g2
```

Now you may import `g2` package of the specified curve

```go
import (
    "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bls254/g2"
)
```

These packages include `G2Projective` and `G2Affine` points as well as a `G2Msm` method.

```go
package main

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254"
	g2 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254/g2"
)

func main() {
	cfg := bn254.GetDefaultMSMConfig()
	size := 1 << 12
	batchSize := 3
	totalSize := size * batchSize
	scalars := bn254.GenerateScalars(totalSize)
	points := g2.G2GenerateAffinePoints(totalSize)

	var p g2.G2Projective
	var out core.DeviceSlice
	out.Malloc(batchSize*p.Size(), p.Size())
	g2.G2Msm(scalars, points, &cfg, out)
}

```

`G2Msm` works the same way as normal MSM, the difference is that it uses G2 Points.
