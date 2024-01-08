# Icicle example: Number-Theoretical Transform (NTT)

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` provides several CUDA C++ template functions for [Number Theoretical Transform](https://github.com/ingonyama-zk/ingopedia/blob/master/src/fft.md), also known as Discrete Fourier Transform. The templates differ in terms of ease-of-use vs. speed. In this example we look a the easiest one.

## Concise Usage Explanation

First include NTT template, next select the curve, and finally supply the curve types to the template. 

```c++
#include "icicle/appUtils/ntt/ntt.cuh"              // template
#include "icicle/curves/bls12_381/curve_config.cuh" // curve
using namespace BLS12_381;
...
ntt_end2end_batch_template<scalar_t, scalar_t>(scalars, batch_size, ntt_size, inverse, stream);
```

In this example we use `BLS12_381` curve. The function computes TODO.


**Parameters:**

TODO


## What's in the example

TODO



