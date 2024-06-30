# MSM Pre computation

To understand the theory behind MSM pre computation technique refer to Niall Emmart's [talk](https://youtu.be/KAWlySN7Hm8?feature=shared&t=1734).

## Core package

### MSM PrecomputePoints

`PrecomputePoints` and `G2PrecomputePoints` exists for all supported curves.

#### Description

This function extends each provided base point $(P)$ with its multiples $(2^lP, 2^{2l}P, ..., 2^{(precompute_factor - 1) \cdot l}P)$, where $(l)$ is a level of precomputation determined by the `precompute_factor`. The extended set of points facilitates faster MSM computations by allowing the MSM algorithm to leverage precomputed multiples of base points, reducing the number of point additions required during the computation.

The precomputation process is crucial for optimizing MSM operations, especially when dealing with large sets of points and scalars. By precomputing and storing multiples of the base points, the MSM function can more efficiently compute the scalar-point multiplications.

#### `PrecomputePoints`

Precomputes points for MSM by extending each base point with its multiples.

```go
func PrecomputePoints(points core.HostOrDeviceSlice, msmSize int, cfg *core.MSMConfig, outputBases core.DeviceSlice) cr.CudaError
```

##### Parameters

- **`points`**: A slice of the original affine points to be extended with their multiples.
- **`msmSize`**: The size of a single msm in order to determine optimal parameters.
- **`cfg`**: The MSM configuration parameters.
- **`outputBases`**: The device slice allocated for storing the extended points.

##### Example

```go
package main

import (
	"log"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254"
)

func main() {
	cfg := bn254.GetDefaultMSMConfig()
	points := bn254.GenerateAffinePoints(1024)
	var precomputeFactor int32 = 8
	var precomputeOut core.DeviceSlice
	precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())

	err := bn254.PrecomputePoints(points, 1024, &cfg, precomputeOut)
	if err != cr.CudaSuccess {
		log.Fatalf("PrecomputeBases failed: %v", err)
	}
}
```

#### `G2PrecomputePoints`

This method is the same as `PrecomputePoints` but for G2 points. Extends each G2 curve base point with its multiples for optimized MSM computations.

```go
func G2PrecomputePoints(points core.HostOrDeviceSlice, msmSize int, cfg *core.MSMConfig, outputBases core.DeviceSlice) cr.CudaError
```

##### Parameters

- **`points`**: A slice of the original affine points to be extended with their multiples.
- **`msmSize`**: The size of a single msm in order to determine optimal parameters.
- **`cfg`**: The MSM configuration parameters.
- **`outputBases`**: The device slice allocated for storing the extended points.

##### Example

```go
package main

import (
	"log"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	g2 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254/g2"
)

func main() {
	cfg := g2.G2GetDefaultMSMConfig()
	points := g2.G2GenerateAffinePoints(1024)
	var precomputeFactor int32 = 8
	var precomputeOut core.DeviceSlice
	precomputeOut.Malloc(points[0].Size()*points.Len()*int(precomputeFactor), points[0].Size())

	err := g2.G2PrecomputePoints(points, 1024, 0, &cfg, precomputeOut)
	if err != cr.CudaSuccess {
		log.Fatalf("PrecomputeBases failed: %v", err)
	}
}
```
