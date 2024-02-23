# Golang Bindings

In order to build the underlying ICICLE libraries you should run the build script `build.sh` found in the `wrappers/golang` directory.

Build script USAGE

```
./build <curve> [G2_enabled]

curve - The name of the curve to build or "all" to build all curves
G2_enabled - Optional - To build with G2 enabled 
```

To build ICICLE libraries for all supported curves with G2 enabled.

```
./build.sh all ON
```

If you wish to build for a specific curve, for example bn254, without G2 enabled.

```
./build.sh bn254
```

>[!NOTE]
>Current supported curves are `bn254`, `bls12_381`, `bls12_377` and `bw6_671`

>[!NOTE]
>G2 is enabled by building your golang project with the build tag `g2`
>Make sure to add it to your build tags if you want it enabled

## Running golang tests

To run the tests for curve bn254.

```
go test ./wrappers/golang/curves/bn254 -count=1
```

To run all the tests in the golang bindings

```
go test --tags=g2 ./... -count=1
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

## Common issues

### Cannot find shared library

In some cases you may encounter the following error, despite exporting the correct `LD_LIBRARY_PATH`.

```
/usr/local/go/pkg/tool/linux_amd64/link: running gcc failed: exit status 1
/usr/bin/ld: cannot find -lbn254: No such file or directory
/usr/bin/ld: cannot find -lbn254: No such file or directory
/usr/bin/ld: cannot find -lbn254: No such file or directory
/usr/bin/ld: cannot find -lbn254: No such file or directory
/usr/bin/ld: cannot find -lbn254: No such file or directory
collect2: error: ld returned 1 exit status
```

This is normally fixed by exporting the path to the shared library location in the following way: `export CGO_LDFLAGS="-L/<path_to_shared_lib>/"`

### cuda_runtime.h: No such file or directory

```
# github.com/ingonyama-zk/icicle/wrappers/golang/curves/bls12381
In file included from wrappers/golang/curves/bls12381/curve.go:5:
wrappers/golang/curves/bls12381/include/curve.h:1:10: fatal error: cuda_runtime.h: No such file or directory
    1 | #include <cuda_runtime.h>
      |          ^~~~~~~~~~~~~~~~
compilation terminated.
```

Our golang bindings rely on cuda headers and require that they can be found as system headers. Make sure to add the `cuda/include` of your cuda installation to your CPATH

```
export CPATH=$CPATH:<path/to/cuda/include>
```
