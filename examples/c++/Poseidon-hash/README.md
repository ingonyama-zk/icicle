# Icicle example: build a Merkle tree using Poseidon hash

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` provides CUDA C++ template `poseidon_hash` to accelerate the popular [Poseidon hash function](https://www.poseidon-hash.info/).

## Concise Usage Explanation

```c++
#include "appUtils/poseidon/poseidon.cu"
...
poseidon_hash<scalar_t, arity+1>(input, output, n, constants, config);
```

**Parameters:**

- **`scalar_t`:** a scalar field of the selected curve.
You can think of field's elements as 32-byte integers modulo `p`, where `p` is a prime number, specific to this field.

- **arity:** number of elements in a hashed block.

- **n:** number of blocks we hash in parallel.

- **input, output:** `scalar_t` arrays of size $arity*n$ and $n$ respectively.

- **constants:** are defined as below

```c++
device_context::DeviceContext ctx= device_context::get_default_device_context();
PoseidonConstants<scalar_t> constants;
init_optimized_poseidon_constants<scalar_t>(ctx, &constants);
```

## What's in the example

1. Define the size of the example: the height of the full binary Merkle tree. 
2. Hash blocks in parallel. The tree width determines the number of blocks to hash.
3. Build a Merkle tree from the hashes.
4. Use the tree to generate a membership proof for one of computed hashes.
5. Validate the hash membership.
6. Tamper the hash.
7. Invalidate the membership of the tempered hash.

## Details

### Merkle tree structure

Our Merkle tree is a **full binary tree** stored in a 1D array.
The tree nodes are stored following a level-first traversal of the binary tree.
For a given level, we use offset to number elements from left to right. The node numbers on the figure below correspond to their locations in the array.

```text
        Tree        Level
          0         0 
        /   \
       1     2      1
      / \   / \
     3   4 5   6    2

1D array representation: {0, 1, 2, 3, 4, 5, 6}
```

### Membership proof structure

We use two arrays:

- position (left/right) of the node along the path toward the root
- hash of a second node with the same parent