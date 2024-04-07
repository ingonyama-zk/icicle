# ICICLE example: computationally-intensive elements of RISC0 protocol

## Why RISC0?

[RISC0 Protocol](https://www.risczero.com/) creates computational integrity proofs (a.k.a. Zero Knowledge Proofs) for programs executing on RISC-V architecture.
The proofs are created for sequences of values in RISC-V registers, called execution traces.
This approach is transparent to developers and enables the use of general purpose languages.

Our [analysis of RISC0 protocol](https://www.ingonyama.com/blog/risc-zero-prover-protocol-analysis) shows that the most computationally-intensive part of the protocol is Fast Reed-Solomon Interactive Oracle Proof (FRI).
Our example will focus on FRI.

## Best-Practices

This example builds on [ICICLE Polynomial API](../polynomial-api/README.md) and [Poseidon hashes](../poseidon/README.md) so we recommend to run them first.

## Key-Takeaway

RISC0 encodes execution traces into very large polynomials and commits them using Merkle trees.
FRI speeds-up validation of such commitments by recursively generating smaller polynomials (and trees) from larger ones.
The key enabler for *recursion* is the *redundancy* of polynomial commitments, hence the use of Reed-Solomon codes.

## What's in the example

This is a **toy** example executing just the first round of FRI

1. Initialize ICICLE: NTT, Polynomials, Poseidon, and Merkle tree
2. Generate random execution trace data
3. Reconstruct polynomial from the trace data
4. Generate Reed-Solomon codeword by interpolating the trace
5. Reconstruct polynomial from the codeword
6. Commit to the codeword polynomial: evaluate at shifted points and encode into Merkle tree
7. FRI Protocol (Commit Phase)
8. FRI Protocol (Query Phase)
