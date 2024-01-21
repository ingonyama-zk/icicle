# Icicle example: build a Merkle tree using Poseidon hash

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

`Icicle` provides CUDA C++ template classes to accelerate Zero Knowledge (ZK) applications, for example, a popular [Poseidon hash function](https://www.poseidon-hash.info/).
Use class `Poseidon` to instantiate and use the hash function


### Instantiate hash function

```c++
Poseidon<BLS12_381::scalar_t> poseidon(arity, stream);
```

**Parameters:**

- **data class:** Here the hash operates on `BLS12_381::scalar_t`, a scalar field of the curve  `BLS12-381`.
You can think of field's elements as 32-bytes integers modulo `p`, where `p` is a prime number, specific to this field.

- **arity:** The number of elements in a hashed block.

- **stream:** CUDA streams allow multiple hashes and higher throughput.

### Hash multiple blocks in parallel

```c++
poseidon.hash_blocks(inBlocks, nBlocks, outHashes, hashType, stream);
```

**Parameters:**

- **nBlocks:** number of blocks we hash in parallel.

- **inBlocks:** input array of size `arity*nBlocks`. The blocks are arranged sequentially in the array.

- **outHashes:** output array of size `nBlocks`.

- **HashType:** In this example we use `Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree`.

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


```
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








