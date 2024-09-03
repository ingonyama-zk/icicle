# Golang bindings

Golang bindings allow you to use ICICLE as a golang library.
The source code for all Golang packages can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang).

The Golang bindings are comprised of multiple packages.

[`core`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/core) which defines all shared methods and structures, such as configuration structures, or memory slices.

[`runtime`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/runtime) which defines abstractions for ICICLE methods for allocating memory, initializing and managing streams, and `Device` which enables users to define and keep track of devices.

Each supported curve and field has its own package which you can find in the respective directories [here](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang). If your project uses BN254 you only need to import that single package named [`bn254`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/curves/bn254).

## Using ICICLE Golang bindings in your project

To add ICICLE to your `go.mod` file.

```bash
go get github.com/ingonyama-zk/icicle/v3
```

If you want to specify a specific branch

```bash
go get github.com/ingonyama-zk/icicle/v3@<branch_name>
```

For a specific commit

```bash
go get github.com/ingonyama-zk/icicle/v3@<commit_id>
```

### Building from source

To build the shared libraries you can run [this](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/build.sh) script inside the downloaded go dependency:

```sh
./build.sh [-curve=<curve>] [-field=<field>] [-cuda_version=<version>] [-skip_msm] [-skip_ntt] [-skip_g2] [-skip_ecntt] [-skip_fieldext]

curve - The name of the curve to build or "all" to build all supported curves
field - The name of the field to build or "all" to build all supported fields
-skip_msm - Optional - build with MSM disabled
-skip_ntt - Optional - build with NTT disabled
-skip_g2 - Optional - build with G2 disabled 
-skip_ecntt - Optional - build with ECNTT disabled
-skip_fieldext - Optional - build without field extension
-help - Optional - Displays usage information
```

:::note

If more than one curve or more than one field is supplied, the last one supplied will be built

:::

To build ICICLE libraries for all supported curves without certain features, you can use their -skip_<feature> flags. For example, for disabling G2 and ECNTT:

```bash
./build.sh -curve=all -skip_g2 -skip_ecntt
```

By default, all features are enabled. To build for a specific field or curve, you can pass the `-field=<field_name>` or `-curve=<curve_name>` flags:

``` bash
./build.sh -curve=bn254
```

Now you can import ICICLE into your project

```go
import (
    "github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
    "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)
...
```

### Building with precompiled libs

Download the frontend release binaries from our [github release page](https://github.com/ingonyama-zk/icicle/releases), for example: icicle30-ubuntu22.tar.gz for ICICLE v3 on ubuntu 22.04

Extract the libs and move them to the downloaded go dependency in your GOMODCACHE

```sh
# extract frontend part
tar xzvf icicle30-ubuntu22.tar.gz
cp -r ./icicle/lib/* $(go env GOMODCACHE)/github.com/ingonyama-zk/icicle/v3@<version>/build/lib/
```

## Running tests

To run all tests, for all curves:

```bash
go test ./... -count=1
```

If you wish to run test for a specific curve or field:

```bash
go test <path_to_curve_or_field> -count=1
```

## How do Golang bindings work?

The golang packages are binded to the libraries produced from compiling ICICLE using cgo.

1. These libraries (named `libicicle_curve_<curve>.a` and `libicicle_field_<curve>.a`) can be imported in your Go project to leverage the accelerated functionalities provided by ICICLE.

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
