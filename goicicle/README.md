# Golang Bindings

To build the shared library:

To build shared libraries for all supported curves.

```
make all
```

If you wish to build for a specific curve, for example bn254.

```
make libbn254.so
```

The current supported options are `libbn254.so`, `libbls12_381.so`, `libbls12_377.so` and `libbw6_671.so`. The resulting `.so` files are the compiled shared libraries for each curve.

Finally to allow your system to find the shared libraries

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/<path_to_shared_libs>
```

## Running golang tests

To run the tests for curve bn254.

```
go test ./goicicle/curves/bn254 -count=1
```

## Cleaning up

If you want to remove the compiled files

```
make clean
```

This will remove all shared libraries generated from the `make` file.

# How do Golang bindings work?

The shared libraries produced from the CUDA code compilation are used to bind Golang to ICICLE's CUDA code.

1. These shared libraries (`libbn254.so`, `libbls12_381.so`, `libbls12_377.so`, `libbw6_671.so`) can be imported in your Go project to leverage the GPU accelerated functionalities provided by ICICLE.

2. In your Go project, you can use `cgo` to link these shared libraries. Here's a basic example on how you can use `cgo` to link these libraries:

```go
/*
#cgo LDFLAGS: -L/path/to/shared/libs -lbn254 -lbls12_381 -lbls12_377 -lbw6_671
#include "icicle.h" // make sure you use the correct header file(s)
*/
import "C"

func main() {
    // Now you can call the C functions from the ICICLE libraries.
    // Note that C function calls are prefixed with 'C.' in Go code.
}
```

Replace `/path/to/shared/libs` with the actual path where the shared libraries are located on your system.

# Common issues

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
