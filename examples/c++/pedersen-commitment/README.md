# ICICLE example: Pedersen Commitment

## Key-Takeaway

A Pedersen Commitment is a cryptographic primitive to commit to a value or a vector of values while keeping it hidden, yet enabling the committer to reveal the value later. It provides both hiding (the commitment does not reveal any information about the value) and binding properties (once a value is committed, it cannot be changed without detection).

Pedersen commitment is based on Multi-Scalar Multiplication [MSM](https://github.com/ingonyama-zk/ingopedia/blob/master/src/msm.md).
`ICICLE` provides CUDA C++ support for [MSM](https://dev.ingonyama.com/icicle/primitives/msm). 
An example of MSM is [here](../msm/README.md).

## Running the example

```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```
## Concise Explanation

We recommend this simple [explanation](https://www.rareskills.io/post/pedersen-commitment).

The original paper: T. P. Pedersen, "Non-Interactive and Information-Theoretic Secure Verifiable Secret Sharing," in Advances in Cryptology — CRYPTO ’91, Lecture Notes in Computer Science, vol 576. Springer, Berlin, Heidelberg.

## What's in the example

1. Define the curve and the size of commitment vector
2. Use public random seed to transparently generate points on the elliptic curve without known discrete logarithm
3. Generate (random) commitment vector and salt (a.k.a blinding factor)
4. Configure and execute MSM using on-host data
5. Output commitment as elliptic point
