# ICICLE example: Number Theoretic Transform (NTT) in Rust

## Key-Takeaway

`ICICLE` provides Golang bindings to CUDA-accelerated C++ implementation of [Number Theoretic Transform](https://github.com/ingonyama-zk/ingopedia/blob/master/src/fft.md).

## Usage

```go
err := Ntt(
  /* input slice */ scalars,
  /* NTT Direction */ core.KForward,
  /* NTT Configuration */ &cfg,
  /* output slice */ result)
```

In this example we use the `BN254` and `BLS12377` fields.

## What's in this example

1. Define the size of NTT.
2. Generate random inputs
3. Set up the domain.
4. Configure NTT
5. Execute NTT on-device
6. Move the result on host

Running the example:

```sh
go run main.go
```

> [!NOTE]
> The default size is 2^20. You can change this by passing the `-s <size>` option. To change the size to 2^23, run the example like this:

```sh
go run main.go -s=23
```