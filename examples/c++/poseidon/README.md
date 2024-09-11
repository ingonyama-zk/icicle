# Icicle example: Number-Theoretical Transform (NTT)

## Key-Takeaway

`Icicle` provides CUDA C++ template function of the Poseidon hash (single hash and multiple hashes).

## Concise Usage Explanation

1. Include the curve api
2. Prepare input, round constants and all the needed matrices (in this example they are loaded from a separate file).
3. Intantiate Poseidon class constructor
4. Call poseidon api (either signle hash or multiple hashes)

```c++
#include "icicle/api/bn254.h"
...
#include "run_single_hash.in_params.h"
...
icicle::Poseidon<scalar_t> poseidon(arity, alpha, nof_partial_rounds, nof_upper_full_rounds, nof_end_full_rounds, rounds_constants, mds_matrix, pre_matrix, sparse_matrices);
...
poseidon.run_single_hash(pre_round_input_state, single_hash_out_limbs, config);
or
poseidon.run_multiple_hash(multiple_hash_in_limbs, multiple_hash_out_limbs, nof_hashes, config));
```


## Running the example

```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```

## What's in the example

1. Define the size of the example
2. Initialize input and constants
3. Run either single hash or multiple hashes
4. Validate the data output
