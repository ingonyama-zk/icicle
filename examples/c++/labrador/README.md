# Labrador protocol

Here we demonstrate a simple implementation of the [Labrador protocol](https://link.springer.com/chapter/10.1007/978-3-031-38554-4_17) using Icicle (also see our [blog](https://hackmd.io/@Ingonyama/fast-labrador-prover)).

To run the program on CPU use

```
./run.sh
```

To run on GPU, first do (only required once)

```
./scripts/pull_cuda_backend.sh
```

then

```
./run.sh -d CUDA
```

The main program runs a simple benchmarking program for which the parameters can be set here:

```cpp
  std::vector<std::tuple<size_t, size_t>> arr_nr{{1 << 6, 1 << 3}};
  std::vector<std::tuple<size_t, size_t>> num_constraint{{10, 10}};
  size_t NUM_REP = 1;
  bool SKIP_VERIF = false;
  benchmark_program(arr_nr, num_constraint, NUM_REP, SKIP_VERIF);
```

You can also run the function `prover_verifier_trace` in main. This function runs 3 recursion iterations of the Labrador protocol for a random instance.

The following flag in [prover.h](./prover.h) can be used to control the program output:

```cpp
// SHOW_STEPS creates a print output listing every step performed by the Prover and the time taken
constexpr bool SHOW_STEPS = true;
```

All functions and objects are documented in code.
