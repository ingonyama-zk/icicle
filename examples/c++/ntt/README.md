# Icicle example: Number-Theoretical Transform (NTT)

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` provides CUDA C++ template function NTT for [Number Theoretical Transform](https://github.com/ingonyama-zk/ingopedia/blob/master/src/fft.md), also known as Discrete Fourier Transform. 

## Concise Usage Explanation

1. Select the curve
2. Include NTT template
3. Configure NTT (TODO)
4. Call NTT

```c++
#define CURVE_ID 1
#include "icicle/appUtils/ntt/ntt.cu"
using namespace curve_config;

...
ntt::NTT<S, E>(input, ntt_size, ntt::NTTDir::kForward, config, output);
```

**Parameters:**

TODO

## What's in the example

TODO



