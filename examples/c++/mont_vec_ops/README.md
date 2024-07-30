#Icicle example : Montgomery vector operations(mul, add, sub) for allpossible options:
is_a_on_device
is_b_on_device
is_result_on_device
is_in_montgomery_form
(is_async isn't checked)

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` accelerates multiplication operation `*` using [Karatsuba algorithm](https://en.wikipedia.org/wiki/Karatsuba_algorithm)

## Concise Usage Explanation

Define field to be used, e. g.:

```c++
#include "api/bn254.h"
```

```c++
using namespace bn254;
typedef scalar_t T;
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
