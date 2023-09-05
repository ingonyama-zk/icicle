# ICICLE CUDA to Golang Binding Guide

This guide provides instructions on how to compile CUDA code using the provided Makefile, and then how to use the resulting shared libraries to bind Golang to ICICLE's CUDA code.

## Prerequisites

- Follow the main prerequisites instructions [here][MAIN_DOCS].
- Golang
- Make

# Compiling and using bindings with Golang

The Makefile is designed to compile CUDA files for three curves: BN254, BLS12_381, and BLS12_377. The source files are located in the `icicle/curves/` directory.

## Compiling CUDA Code

1. Navigate to the directory containing the Makefile in your terminal.
2. To compile all curve libraries, use the `make all` command. This will create three shared libraries: `libbn254.so`, `libbls12_381.so`, and `libbls12_377.so`.
3. If you want to compile a specific curve, you can do so by specifying the target. For example, to compile only the BN254 curve, use `make libbn254.so`. Replace `libbn254.so` with `libbls12_381.so` or `libbls12_377.so` to compile those curves instead.
4. run `export LD_LIBRARY_PATH=<path_to_so_files_directory>/`, this will make the shared libraries available. 

The resulting `.so` files are the compiled shared libraries for each curve.

If you want to remove the compiled files, you can use the `make clean` command. This will remove the `libbn254.so`, `libbls12_381.so`, and `libbls12_377.so` files.

## Using GOCICLE

We assume you have compiled the shared libraries (process described above) and have them on your machine.

1. Install the GOICICLE [package](https://pkg.go.dev/github.com/ingonyama-zk/icicle/goicicle). Using this command `go get github.com/ingonyama-zk/icicle/goicicle`.
2. Make sure you called `export LD_LIBRARY_PATH=<path_to_so_files_directory>/` to make your shared libraries available.
3. Import GOICICLE into your project and enjoy :)

For reference have a look at our ICICLE <> Gnark [implementation][GNARKI].

## Using the bindings without GOICICLE

The shared libraries produced from the CUDA code compilation are used to bind Golang to ICICLE's CUDA code.

1. These shared libraries (`libbn254.so`, `libbls12_381.so`, `libbls12_377.so`) can be imported in your Go project to leverage the GPU accelerated functionalities provided by ICICLE. 

2. In your Go project, you can use `cgo` to link these shared libraries. Here's a basic example on how you can use `cgo` to link these libraries:

```go
/*
#cgo LDFLAGS: -L/path/to/shared/libs -lbn254 -lbls12_381 -lbls12_377
#include "icicle.h" // make sure you use the correct header file(s)
*/
import "C"

func main() {
    // Now you can call the C functions from the ICICLE libraries.
    // Note that C function calls are prefixed with 'C.' in Go code.
}
```

Replace `/path/to/shared/libs` with the actual path where the shared libraries are located on your system.

## Running tests

Running ``go test ./...`` in the root directory should execute all tests.

If you want to run a specific test for a specific curve, navigate to `.goicicle/curves/<curve_name>`. Then run `go test -run=<test_name>`

I suggest adding `-v` to see more verbose logs when running a test.

# Supporting Additional Curves

Before adding support for a curve in the golang bindings you must add it to the CUDA code.

1. Add your new curve to CUDA ICICLE, by following the instructions [here][MAIN_DOCS].
2. Add the correct template for your curve [here][GOICICLE_CURVE_TEMP]. It should look like this:

```
type Curve struct {
	PackageName        string
	CurveNameUpperCase string
	CurveNameLowerCase string
	SharedLib          string
	ScalarSize         int
	BaseSize           int
	G2ElementSize      int
}
```
3. Edit this [file][GOICICLE_CURVE_FILE_TO_EDIT] accordingly.
4. navigate to `goicicle/templates` and then run `go run main.go`.

*** Make sure to compile your new shared libraries :)

## Publishing your curve

We suggest you open a PR with your new curve so we can add official support for it.
In the meantime if you wish to use this curve simple push it to your forked ICICLE repository, then install it with the following command: `go get <github_com_path_to_github_repo>@<branch>`.


<!-- Begin Links -->
[MAIN_DOCS]: ./README.md
[GOICICLE_CURVE_TEMP]: ./templates/curves/curves.go
[GOICICLE_CURVE_FILE_TO_EDIT]: ./templates/main.go
[GNARKI]: https://github.com/ingonyama-zk/gnark
<!-- End Links -->
