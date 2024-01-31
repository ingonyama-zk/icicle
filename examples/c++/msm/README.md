# Icicle example: Muli-Scalar Multiplication (MSM)

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` provides CUDA C++ template function `MSM` to accelerate [Multi-Scalar Multiplication](https://github.com/ingonyama-zk/ingopedia/blob/master/src/msm.md).

## Concise Usage Explanation

1. Select the curve
2. Include an MSM template
3. Configure MSM
4. Call the template  

```c++
#define CURVE_ID 1
#include "icicle/appUtils/msm/msm.cu"
...
msm::MSMConfig config = {...};
...
msm::MSM<scalar_t, affine_t, projective_t>(scalars, points, size, config, &result);
```

In this example we use `BN254` curve (`CURVE_ID=1`). The function computes $result = \sum_{i=0}^{size-1} scalars[i] \cdot points[i]$, where input `points[]` use affine coordinates, and `result` uses projective coordinates.

**Parameters:**

The configuration is passed to the kernel as a structure of type `msm::MSMConfig`. Some of the most important fields are listed below:

- `are_scalars_on_device`, `are_points_on_device`, `are_results_on_device`: location of the data

- `is_async`: blocking vs. non-blocking kernel call

- `large_bucket_factor`:  distinguishes between large bucket and normal bucket sizes. If there is a scalar distribution that is skewed heavily to a few values we can operate on those separately from the rest of the values. The ideal value here can vary by circuit (based on the distribution of scalars) but start with 10 and adjust to see if it improves performance.

## Running the example

- `cd` to your example directory
- compile with  `./compile.sh`
- run with `./run.sh`

## What's in the example

1. Define the parameters of MSM
2. Generate random inputs on-host
3. Configure and execute MSM using on-host data
4. Copy inputs on-device
5. Configure and execute MSM using on-device data
6. Repeat the above steps for G2 points
