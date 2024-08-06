# ICICLE example: RISC0's Fibonacci sequence proof using Polynomial API

## Why RISC0?

[RISC0 Protocol](https://www.risczero.com/) creates computational integrity proofs (a.k.a. Zero Knowledge Proofs) for programs executing on RISC-V architecture.
The proofs are created for sequences of values in RISC-V registers, called execution traces.
This approach is transparent to developers and enables the use of general purpose languages.

## Best-Practices

This example builds on [ICICLE Polynomial API](../polynomial-api/README.md) so we recommend to run it first.

## Key-Takeaway

RISC0 encodes execution traces into very large polynomials and commits them using Merkle trees.
FRI speeds-up validation of such commitments by recursively generating smaller polynomials (and trees) from larger ones.
The key enabler for *recursion* is the *redundancy* of polynomial commitments, hence the use of Reed-Solomon codes.

## Running the example

To run example, from project root directory:

```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```

## What's in the example

The example follows [STARK by Hand](https://dev.risczero.com/proof-system/stark-by-hand), structured in the following Lessons:

1. The Execution Trace
2. Rule checks to validate a computation
3. Padding the Trace
4. Constructing Trace Polynomials
5. ZK Commitments of the Trace Data
6. Constraint Polynomials
7. Mixing Constraint Polynomials
8. The Core of the RISC Zero STARK
9. The DEEP Technique
10. Mixing (Batching) for FRI
11. FRI Protocol (Commit Phase)
12. FRI Protocol (Query Phase)
