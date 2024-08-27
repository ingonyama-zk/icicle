# Golang bindings

TODO update for V3

Golang bindings allow you to use ICICLE as a golang library.
The source code for all Golang packages can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang).

The Golang bindings are comprised of multiple packages.

[`core`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/core) which defines all shared methods and structures, such as configuration structures, or memory slices.

[`cuda-runtime`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/cuda_runtime) which defines abstractions for CUDA methods for allocating memory, initializing and managing streams, and `DeviceContext` which enables users to define and keep track of devices.

Each supported curve, field, and hash has its own package which you can find in the respective directories [here](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang). If your project uses BN254 you only need to import that single package named [`bn254`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/curves/bn254).

## Using ICICLE Golang bindings in your project

To add ICICLE to your `go.mod` file.

```bash
go get github.com/ingonyama-zk/icicle
```

If you want to specify a specific branch

```bash
go get github.com/ingonyama-zk/icicle@<branch_name>
```

For a specific commit

```bash
go get github.com/ingonyama-zk/icicle@<commit_id>
```

To build the shared libraries you can run [this](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/build.sh) script:

```sh
./build.sh [-curve=<curve>] [-field=<field>] [-hash=<hash>] [-cuda_version=<version>] [-g2] [-ecntt] [-devmode]

curve - The name of the curve to build or "all" to build all supported curves
field - The name of the field to build or "all" to build all supported fields
hash - The name of the hash to build or "all" to build all supported hashes
-g2 - Optional - build with G2 enabled 
-ecntt - Optional - build with ECNTT enabled
-devmode - Optional - build in devmode
-help - Optional - Displays usage information
```

:::note

If more than one curve or more than one field or more than one hash is supplied, the last one supplied will be built

:::

To build ICICLE libraries for all supported curves with G2 and ECNTT enabled.

```bash
./build.sh -curve=all -g2 -ecntt
```

If you wish to build for a specific curve, for example bn254, without G2 or ECNTT enabled.

``` bash
./build.sh -curve=bn254
```

Now you can import ICICLE into your project

```go
import (
    "github.com/stretchr/testify/assert"
    "testing"

    "github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
    cr "github.com/ingonyama-zk/icicle/v3/wrappers/golang/cuda_runtime"
)
...
```

## Running tests

To run all tests, for all curves:

```bash
go test ./... -count=1
```

If you wish to run test for a specific curve:

```bash
go test <path_to_curve> -count=1
```

## How do Golang bindings work?

The libraries produced from the CUDA code compilation are used to bind Golang to ICICLE's CUDA code.

1. These libraries (named `libingo_curve_<curve>.a` and `libingo_field_<curve>.a`) can be imported in your Go project to leverage the GPU accelerated functionalities provided by ICICLE.

2. In your Go project, you can use `cgo` to link these libraries. Here's a basic example on how you can use `cgo` to link these libraries:

```go
/*
#cgo LDFLAGS: -L/path/to/shared/libs -lingo_curve_bn254 -L$/path/to/shared/libs -lingo_field_bn254 -lstdc++ -lm
#include "icicle.h" // make sure you use the correct header file(s)
*/
import "C"

func main() {
    // Now you can call the C functions from the ICICLE libraries.
    // Note that C function calls are prefixed with 'C.' in Go code.
}
```

Replace `/path/to/shared/libs` with the actual path where the shared libraries are located on your system.

## Supported curves, fields and operations

### Supported curves and operations

| Operation\Curve | bn254 | bls12_377 | bls12_381 | bw6-761 | grumpkin |
| --------------- | :---: | :-------: | :-------: | :-----: | :------: |
| MSM             |   ✅   |     ✅     |     ✅     |    ✅    |    ✅     |
| G2              |   ✅   |     ✅     |     ✅     |    ✅    |    ❌     |
| NTT             |   ✅   |     ✅     |     ✅     |    ✅    |    ❌     |
| ECNTT           |   ✅   |     ✅     |     ✅     |    ✅    |    ❌     |
| VecOps          |   ✅   |     ✅     |     ✅     |    ✅    |    ✅     |
| Polynomials     |   ✅   |     ✅     |     ✅     |    ✅    |    ❌     |

### Supported fields and operations

| Operation\Field | babybear |
| --------------- | :------: |
| VecOps          |    ✅     |
| Polynomials     |    ✅     |
| NTT             |    ✅     |
| Extension Field |    ✅     |
