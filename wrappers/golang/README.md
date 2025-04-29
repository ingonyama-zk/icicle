# Go Bindings

In order to build the underlying ICICLE libraries you should run the build script found [here](./internal/build-libs.go).

Build script USAGE

```sh
go run build-libs.go [-curve=<curve,curve2>] [-field=<field,field2>] [-hash=<hash>] [-cuda_version=<version>] [-skip_msm] [-skip_ntt] [-skip_g2] [-skip_ecntt] [-skip_fieldext]

-curve <string>           Comma-separated list of curves to be built. If "all" is supplied,
                            all curves will be built with any additional curve options.
-field <string>           Comma-separated list of fields to be built. If "all" is supplied,
                            all fields will be built with any additional field options
-skip-msm                 Builds the curve library with MSM disabled
-skip-ntt                 Builds the curve/field library with NTT disabled
-skip-g2                  Builds the curve library with G2 disabled
-skip-ecntt               Builds the curve library with ECNTT disabled
-skip-poseidon            Builds the curve or field library with poseidon hashing disabled
-skip-poseidon2           Builds the curve or field library with poseidon2 hashing disabled
-skip-hash                Builds the library with Hashes disabled
-skip-fieldext            Builds the field library with the extension field disabled
-cuda <string>            Specifies the branch/commit for CUDA backend, or "local"
-cuda-version <string>    Specifies the version of CUDA to use
-metal <string>           Specifies the branch/commit for METAL backend, or "local"
-install-path <string>    Installation path for built libraries
```

To build ICICLE libraries for all supported curves with G2 and ECNTT enabled.

```sh
go run build-libs.go -curve=all
```

If you wish to build for a specific curve, for example bn254, without G2 or ECNTT enabled.

```sh
go run build-libs.go -curve=bn254 -skip-g2 -skip-ecntt
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

## Using the Go frontend

In order to use the Go frontend packages, a few environment variables need to be set:

1. Export the locations where the libraries are installed to the `CGO_LDFLAGS` environment variable.

    ```bash
    export CGO_LDFLAGS="-L</path/to/installed/libs> -Wl,-rpath,<path/to/installed/libs>"
    ```

2. In order to load a backend other than CPU, set the `ICICLE_BACKEND_INSTALL_DIR` environment variable to the location wheren the backend libraries are found

    ```bash
    export ICICLE_BACKEND_INSTALL_DIR="</path/to/installed/backend/libs>"
    ```

3. If you are building for a machine with non-amd architecture, make sure to change the `GOARCH` env var to the correct architecture. For example, when building for Macs with Apple Silicon (using the Metal backend):

    ```bash
    go env -w GOARCH="arm64"
    ```

4. Ensure CGO is enabled by setting the go env variable

   ```bash
   go env -w CGO_ENABLED=1
   ```

## Running Go tests

To run the tests for curve bn254.

```sh
go test ./wrappers/golang/curves/bn254/tests -count=1 -v
```

To run all the tests in the Go bindings

```sh
go test ./... -count=1 -v
```

## How do Go bindings work?

The libraries produced from the code compilation are used to bind the Go frontend to ICICLE's code.

1. These libraries (named `libicicle_curve_<curve>.so` and `libicicle_field_<field>.so`) can be linked to your Go project to leverage the accelerated functionalities provided by ICICLE.

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
