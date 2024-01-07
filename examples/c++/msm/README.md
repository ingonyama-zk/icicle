# Icicle example: Muli-Scalar Multiplication (MSM)

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` provides CUDA C++ template function `large_msm` to accelerate [Multi-Scalar Multiplication](https://github.com/ingonyama-zk/ingopedia/blob/master/src/msm.md).

## Concise Usage Explanation

First include MSM template, next select the curve, and finally supply the curve types to the template. 

```c++
#include "appUtils/msm/msm.cu"           // template
#include "curves/bn254/curve_config.cuh" // curve
using namespace BN254;
...
large_msm<scalar_t, projective_t, affine_t>(scalars,points,size,result,on_device,big_triangle,bucket_factor,stream)
```

In this example we use `BN254` curve. The function computes $result = \sum_{i=0}^{size-1} scalars[i] \cdot points[i]$, where input `points[]` use affine coordinates, and `result` uses projective coordinates.


**Parameters:**

- `on_device`: `true` when executed on GPU, otherwise on host

- `big_triangle`: Depreciated. Always set to `false`.

- `bucket_factor`:  distinguishes between large bucket and normal bucket sizes. If there is a scalar distribution that is skewed heavily to a few values we can operate on those separately from the rest of the values. The ideal value here can vary by circuit (based on the distribution of scalars) but start with 10 and adjust to see if it improves performance.

- `stream`: CUDA streams enable parallel execution of multiple functions

## What's in the example

1. Define the parameters of MSM. 
2. Generate random inputs on-host
3. Copy inputs on-device
4. Execute MSM on-device (GPU)
5. Copy results on-host







