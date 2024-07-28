# Icicle example: Muli-Scalar Multiplication (MSM)

## Key-Takeaway

`Icicle` provides CUDA C++ template function `MSM` to accelerate [Multi-Scalar Multiplication](https://github.com/ingonyama-zk/ingopedia/blob/master/src/msm.md).

## Concise Usage Explanation

1. Include the curve api
2. Configure MSM
3. Call msm api

```c++
#include "icicle/api/bn254.h"
...
MSMConfig config = default_msm_config();
...
bn254_msm(scalars, points, size, config, &result);
```

In this example we use `BN254` curve. The function computes $result = \sum_{i=0}^{size-1} scalars[i] \cdot points[i]$, where input `points[]` use affine coordinates, and `result` uses projective coordinates.

**Parameters:**

The configuration is passed to the kernel as a structure of type `MSMConfig`. Some of the most important fields are listed below:

- `are_scalars_on_device`, `are_points_on_device`, `are_results_on_device`: location of the data

- `is_async`: blocking vs. non-blocking kernel call

- In addition can pass backend-specific params via config.extConfig. For example CUDA backend accepts a `large_bucket_factor` param.

## Running the example

```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```

## What's in the example

1. Define the parameters of MSM
2. Generate random inputs on-host
3. Configure and execute MSM using on-host data
4. Copy inputs on-device
5. Configure and execute MSM using on-device data
6. Repeat step 3 G2 msm points
