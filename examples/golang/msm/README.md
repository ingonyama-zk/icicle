# ICICLE example: MultiScalar Multiplication (MSM) in Golang

`ICICLE` provides Golang bindings to CUDA-accelerated C++ implementation of [Multi-Scalar Multiplication](https://github.com/ingonyama-zk/ingopedia/blob/master/src/msm.md).

## Usage

```go
err := Msm(
  /* Scalars input vector */ scalars,
  /* Points input vector */ points,
  /* MSMConfig reference */ &cfg,
  /* Projective point result */ results)
```

In this example we use `BN254` and `BLS12377` curves. The function computes $result = \sum_{i=0}^{size-1} scalars[i] \cdot points[i]$, where input `points[]` uses affine coordinates, and `result` uses projective coordinates.

## What's in the example

1. Define the size of MSM. 
2. Generate random inputs on-device
3. Configure MSM
4. Execute MSM on-device
5. Move the result on host

Running the example:
```sh
go run main.go
```

> [!NOTE]
> The default sizes are 2^17 - 2^22. You can change this by passing the `-l <size> -u <size>` options. To change the size range to 2^21 - 2^24, run the example like this:
> ```sh
> go run main.go -l=21 -u=24
> ```
