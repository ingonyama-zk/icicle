# Icicle example: Number-Theoretical Transform (NTT)

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` provides CUDA C++ template function NTT for [Number Theoretical Transform](https://github.com/ingonyama-zk/ingopedia/blob/master/src/fft.md), also known as Discrete Fourier Transform.

## Concise Usage Explanation

```c++
// Select the curve
#define CURVE_ID 1
// Include NTT template
#include "appUtils/ntt/ntt.cu"
using namespace curve_config;
using namespace ntt;
// Configure NTT
NTTConfig<S> config=DefaultNTTConfig<S>();
// Call NTT
NTT<S, E>(input, ntt_size, NTTDir::kForward, config, output);
```

## Running the example

- `cd` to your example directory
- compile with  `./compile.sh`
- run with `./run.sh`

## What's in the example

1. Define the size of the example
2. Initialize input
3. Run Radix2 NTT
4. Run MexedRadix NTT
5. Validate the data output
