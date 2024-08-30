# Golang Bindings

In order to build the underlying ICICLE libraries you should run the build script found [here](./build.sh).

Build script USAGE

```sh
./build.sh [-curve=<curve>] [-field=<field>] [-hash=<hash>] [-cuda_version=<version>] [-skip_msm] [-skip_ntt] [-skip_g2] [-skip_ecntt] [-skip_fieldext]

curve - The name of the curve to build or "all" to build all supported curves
field - The name of the field to build or "all" to build all supported fields
-skip_msm - Optional - build with MSM disabled
-skip_ntt - Optional - build with NTT disabled
-skip_g2 - Optional - build with G2 disabled 
-skip_ecntt - Optional - build with ECNTT disabled
-skip_fieldext - Optional - build without field extension
-help - Optional - Displays usage information
```

To build ICICLE libraries for all supported curves with G2 and ECNTT enabled.

```sh
./build.sh -curve=all
```

If you wish to build for a specific curve, for example bn254, without G2 or ECNTT enabled.

```sh
./build.sh -curve=bn254 -skip_g2 -skip_ecntt
```

## Supported curves, fields and operations

### Supported curves and operations

| Operation\Curve | bn254 | bls12_377 | bls12_381 | bw6-761 | grumpkin |
| --- | :---: | :---: | :---: | :---: | :---: |
| MSM | ✅ | ✅ | ✅ | ✅ | ✅ |
| G2  | ✅ | ✅ | ✅ | ✅ | ❌ |
| NTT | ✅ | ✅ | ✅ | ✅ | ❌ |
| ECNTT | ✅ | ✅ | ✅ | ✅ | ❌ |
| VecOps | ✅ | ✅ | ✅ | ✅ | ✅ |
| Polynomials | ✅ | ✅ | ✅ | ✅ | ❌ |

### Supported fields and operations

| Operation\Field | babybear |
| --- | :---: |
| VecOps | ✅ |
| Polynomials | ✅ |
| NTT | ✅ |
| Extension Field | ✅ |

## Running golang tests

To run the tests for curve bn254.

```sh
go test ./wrappers/golang_v3/curves/bn254/tests -count=1 -v
```

To run all the tests in the golang bindings

```sh
go test ./... -count=1 -v
```

## How do Golang bindings work?

The libraries produced from the CUDA code compilation are used to bind Golang to ICICLE's CUDA code.

1. These libraries (named `libingo_curve_<curve>.a` and `libingo_field_<curve>.a`) can be imported in your Go project to leverage the GPU accelerated functionalities provided by ICICLE.

2. In your Go project, you can use `cgo` to link these libraries. Here's a basic example on how you can use `cgo` to link these libraries:

```go
/*
#cgo LDFLAGS: -L/path/to/shared/libs -licicle_device -lstdc++ -lm -Wl,-rpath=/path/to/shared/libs
#include "icicle.h" // make sure you use the correct header file(s)
*/
import "C"

func main() {
    // Now you can call the C functions from the ICICLE libraries.
    // Note that C function calls are prefixed with 'C.' in Go code.
}
```

Replace `/path/to/shared/libs` with the actual path where the shared libraries are located on your system.

## Common issues

### cuda_runtime.h: No such file or directory

```sh
In file included from wrappers/golang_v3/curves/bls12381/curve.go:5:
wrappers/golang_v3/curves/bls12381/include/curve.h:1:10: fatal error: cuda_runtime.h: No such file or directory
    1 | #include <cuda_runtime.h>
      |          ^~~~~~~~~~~~~~~~
compilation terminated.
```

Our golang bindings rely on cuda headers and require that they can be found as system headers. Make sure to add the `cuda/include` of your cuda installation to your CPATH

```sh
export CPATH=$CPATH:<path/to/cuda/include>
```
