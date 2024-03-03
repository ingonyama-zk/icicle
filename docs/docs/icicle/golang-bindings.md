# Golang bindings

Golang bindings allow you to use ICICLE as a golang library.
The source code for all Golang libraries can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang).

The Golang bindings are comprised of multiple packages.

[`core`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/core) which defines all common shared methods and structures, such as configuration structures, or memory slices.

[`cuda-runtime`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/cuda_runtime) which defines abstractions for CUDA methods for allocating memory, initializing and managing streams. As well as `DeviceContext` which enables users to define and keep track of devices.

Each curve has its own package, you can find all curves [here](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/curves). If you project uses BN-254 you only need to install that single package named [`bn254`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/curves/bn254).

## Using ICICLE Golang bindings in your project

Too add ICICLE to your `go.mod` file.

```bash
go get github.com/ingonyama-zk/icicle/goicicle
```

If you want to specify a specific branch

```bash
go get github.com/ingonyama-zk/icicle/goicicle@<branch_name>
```

For a specific commit

```bash
go get github.com/ingonyama-zk/icicle/goicicle@<commit_id>
```

To build the shared libraries you can run this script:

```
./build <curve> [G2_enabled]

curve - The name of the curve to build or "all" to build all curves
G2_enabled - Optional - To build with G2 enabled 
```

For example if you want to build all curves with G2 enabled you would run:

```bash
./build.sh all ON
```

If you are interested in building a specific curve you would run:

```bash
./build.sh bls12381 ON
```

After building your shared libraries. You must export them so your system will be aware of their existence.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/<path_to_shared_libs>
```

Now you can ICICLE into your project

```golang
import (
    "github.com/stretchr/testify/assert"
    "testing"

    "github.com/ingonyama-zk/icicle/wrappers/golang/core"
    cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)
...
```

## Running tests

To run all test, for all curves:

```bash
go test --tags=g2 ./... -count=1
```

If you dont want to include g2 tests then drop `--tags=g2`.

If you wish to run test for a specific curve:

```bash
go test <path_to_curve> -count=1
```

## How do Golang bindings work?

The libraries produced from the CUDA code compilation are used to bind Golang to ICICLE's CUDA code.

1. These libraries (named `libingo_<curve>.a`) can be imported in your Go project to leverage the GPU accelerated functionalities provided by ICICLE.

2. In your Go project, you can use `cgo` to link these libraries. Here's a basic example on how you can use `cgo` to link these libraries:

```go
/*
#cgo LDFLAGS: -L/path/to/shared/libs -lingo_bn254
#include "icicle.h" // make sure you use the correct header file(s)
*/
import "C"

func main() {
    // Now you can call the C functions from the ICICLE libraries.
    // Note that C function calls are prefixed with 'C.' in Go code.
}
```

Replace `/path/to/shared/libs` with the actual path where the shared libraries are located on your system.
