# MSM Pre computation

To understand the theory behind MSM pre computation technique refer to Niall Emmart's [talk](https://youtu.be/KAWlySN7Hm8?feature=shared&t=1734).

## Core package

### MSM PrecomputeBases

`PrecomputeBases` and `G2PrecomputeBases` exists for all supported curves.

#### Description

This function extends each provided base point $(P)$ with its multiples $(2^lP, 2^{2l}P, ..., 2^{(precompute_factor - 1) \cdot l}P)$, where $(l)$ is a level of precomputation determined by the `precompute_factor`. The extended set of points facilitates faster MSM computations by allowing the MSM algorithm to leverage precomputed multiples of base points, reducing the number of point additions required during the computation.

The precomputation process is crucial for optimizing MSM operations, especially when dealing with large sets of points and scalars. By precomputing and storing multiples of the base points, the MSM function can more efficiently compute the scalar-point multiplications.

#### `PrecomputeBases`

Precomputes points for MSM by extending each base point with its multiples.

```go
func PrecomputeBases(bases core.HostOrDeviceSlice, cfg *core.MSMConfig, outputBases core.DeviceSlice) runtime.EIcicleError
```

##### Parameters

- **`bases`**: A slice of the original affine points to be extended with their multiples.
- **`cfg`**: The MSM configuration parameters.
- **`outputBases`**: The device slice allocated for storing the extended points.

##### Example

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
	// Load backend using env path
	runtime.LoadBackendFromEnvOrDefault()
	// Set Cuda device to perform
	device := runtime.CreateDevice("CUDA", 0)
	runtime.SetDevice(&device)

	cfg := core.GetDefaultMSMConfig()
	points := bn254.GenerateAffinePoints(1024)
	cfg.PrecomputeFactor = 8
	var precomputeOut core.DeviceSlice
	precomputeOut.Malloc(points[0].Size(), points.Len()*int(cfg.PrecomputeFactor))

	err := msm.PrecomputeBases(points, &cfg, precomputeOut)
	if err != runtime.Success {
		log.Fatalf("PrecomputeBases failed: %v", err)
	}
}
```

#### `G2PrecomputeBases`

This method is the same as `PrecomputePoints` but for G2 points. Extends each G2 curve base point with its multiples for optimized MSM computations.

```go
func G2PrecomputeBases(bases core.HostOrDeviceSlice, cfg *core.MSMConfig, outputBases core.DeviceSlice) runtime.EIcicleError
```

##### Parameters

- **`bases`**: A slice of the original affine points to be extended with their multiples.
- **`cfg`**: The MSM configuration parameters.
- **`outputBases`**: The device slice allocated for storing the extended points.

##### Example

```go
package main

import (
	"log"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/g2"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func main() {
	// Load backend using env path
	runtime.LoadBackendFromEnvOrDefault()
	// Set Cuda device to perform
	device := runtime.CreateDevice("CUDA", 0)
	runtime.SetDevice(&device)

	cfg := core.GetDefaultMSMConfig()
	points := g2.G2GenerateAffinePoints(1024)
	cfg.PrecomputeFactor = 8
	var precomputeOut core.DeviceSlice
	precomputeOut.Malloc(points[0].Size(), points.Len()*int(cfg.PrecomputeFactor))

	err := g2.G2PrecomputeBases(points, &cfg, precomputeOut)
	if err != runtime.Success {
		log.Fatalf("PrecomputeBases failed: %v", err)
	}
}
```
