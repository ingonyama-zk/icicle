# Icicle example: Multiplication

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` accelerates multiplication operation `*` using [Karatsuba algorithm](https://en.wikipedia.org/wiki/Karatsuba_algorithm)

## Concise Usage Explanation

Define a `CURVE_ID` and include curve configuration header:

```c++
#define CURVE_ID 1
#include "curves/curve_config.cuh"
```

The values of `CURVE_ID` for different curves are in the above header. Multiplication is accelerated both for field scalars and point fields.

```c++
using namespace curve_config;
scalar_t a;
point_field_t b;
```

## Running the example

- `cd` to your example directory
- compile with `./compile.sh`
- run with `./run.sh`

## What's in the example

1. Define the parameters for the example such as vector size 
2. Generate random vectors on-host
3. Copy them on-device
4. Execute element-wise vector multiplication on-device
5. Copy results on-host

