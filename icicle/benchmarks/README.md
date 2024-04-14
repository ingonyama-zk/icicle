# How to use benchmarks

ICICLE uses [google benchmarks](https://github.com/google/benchmark) to measure the performance of primitives.

To run benchmarks, make sure you have everything installed to run ICICLE (see top-level README for that). Next, you need to install google benchmarks library as described in their [documentation](https://github.com/google/benchmark?tab=readme-ov-file#installation). When running benchmarks, export the path to this installation:

```
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:<path-to-google-benchmarks-build-folder>
```

Then to benchmark field arithmetic, say, on `babybear` field, run:

```
cmake -UCURVE -UFIELD -UG2 -UEXT_FIELD -DFIELD=babybear -DEXT_FIELD=ON -S . -B build;
cmake --build build;
build/benches --benchmark_counters_tabular=true
```

`-U` parameters are needed to clear variables from previous runs and `EXT_FIELD` can be disabled if benhcmarking the extension field is not needed. To benchmark a curve, say, `bn254`, change the first `cmake` call to:

```
cmake -UCURVE -UFIELD -UG2 -UEXT_FIELD -DCURVE=bn254 -S . -B build;
```

Benchmarks measure throughput of very cheap operations like field multiplication or EC addition by repeating them very many times in parallel, so throughput is the main metric to look at.