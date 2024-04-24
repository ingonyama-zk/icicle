# MSM Pre computation

To understand the theory behind MSM pre computation technique refer to Niall Emmart's [talk](https://youtu.be/KAWlySN7Hm8?feature=shared&t=1734).

### Supported curves

`bls12-377`, `bls12-381`, `bn254`, `bw6-761`, `grumpkin`

## Core package

## MSM `PrecomputeBases`

`PrecomputeBases` and `G2PrecomputeBases` exists for all supported curves. 

#### Description

This function extends each provided base point $(P)$ with its multiples $(2^lP, 2^{2l}P, ..., 2^{(precompute_factor - 1) \cdot l}P)$, where $(l)$ is a level of precomputation determined by the `precompute_factor`. The extended set of points facilitates faster MSM computations by allowing the MSM algorithm to leverage precomputed multiples of base points, reducing the number of point additions required during the computation.

The precomputation process is crucial for optimizing MSM operations, especially when dealing with large sets of points and scalars. By precomputing and storing multiples of the base points, the MSM function can more efficiently compute the scalar-point multiplications.

#### `PrecomputeBases`

Precomputes bases for MSM by extending each base point with its multiples.

```go
func PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, c int32, ctx *cr.DeviceContext, outputBases core.DeviceSlice) cr.CudaError
```

##### Parameters

- **`points`**: A slice of the original affine points to be extended with their multiples.
- **`precomputeFactor`**: Determines the total number of points to precompute for each base point.
- **`c`**: Currently unused; reserved for future compatibility.
- **`ctx`**: CUDA device context specifying the execution environment.
- **`outputBases`**: The device slice allocated for storing the extended bases.

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

	err := bn254.PrecomputeBases(points, precomputeFactor, 0, &cfg.Ctx, precomputeOut)
	if err != cr.CudaSuccess {
		log.Fatalf("PrecomputeBases failed: %v", err)
	}
}
```

#### `G2PrecomputeBases`

This method is the same as `PrecomputeBases` but for G2 points. Extends each G2 curve base point with its multiples for optimized MSM computations.

```go
func G2PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, c int32, ctx *cr.DeviceContext, outputBases core.DeviceSlice) cr.CudaError
```

##### Parameters

- **`points`**: A slice of G2 curve points to be extended.
- **`precomputeFactor`**: The total number of points to precompute for each base.
- **`c`**: Reserved for future use to ensure compatibility with MSM operations.
- **`ctx`**: Specifies the CUDA device context for execution.
- **`outputBases`**: Allocated device slice for the extended bases.

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

	err := g2.G2PrecomputeBases(points, precomputeFactor, 0, &cfg.Ctx, precomputeOut)
	if err != cr.CudaSuccess {
		log.Fatalf("PrecomputeBases failed: %v", err)
	}
}
```

### Benchmarks

Benchmarks where performed on a Nvidia RTX 3090Ti.

| Pre-computation factor | bn254 size `2^20` MSM, ms.  | bn254 size `2^12` MSM, size `2^10` batch, ms. | bls12-381 size `2^20` MSM, ms. | bls12-381 size `2^12` MSM, size `2^10` batch, ms. |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  | 14.1  | 82.8  | 25.5  | 136.7  |
| 2  | 11.8  | 76.6  | 20.3  | 123.8  |
| 4  | 10.9  | 73.8  | 18.1  | 117.8  |
| 8  | 10.6  | 73.7  | 17.2  | 116.0  |
