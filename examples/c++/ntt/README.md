# Icicle example: Number-Theoretical Transform (NTT)

## Key-Takeaway

`Icicle` provides CUDA C++ template function NTT for [Number Theoretical Transform](https://github.com/ingonyama-zk/ingopedia/blob/master/src/fft.md), also known as Discrete Fourier Transform.

## Concise Usage Explanation

1. Include the curve api
2. Init NTT domain
3. Call ntt api

```c++
#include "icicle/api/bn254.h"
...
auto ntt_init_domain_cfg = default_ntt_init_domain_config();
...
bn254_ntt_init_domain(&basic_root, ntt_init_domain_cfg);
NTTConfig<scalar_t> config = default_ntt_config<scalar_t>();
...
bn254_ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get())
```


## Running the example

- `cd` to your example directory
- compile with  `./compile.sh`
- run with `./run.sh`

## What's in the example

1. Define the size of the example
2. Initialize input
3. Run Radix2 NTT
4. Run MixedRadix NTT
5. Validate the data output
