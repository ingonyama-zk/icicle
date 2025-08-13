# Lattice SNARK Example — Types and APIs

This Rust example showcases core types and APIs used in lattice-based SNARK constructions.  
It serves as both a reference and a playground for key abstractions such as polynomial rings, number-theoretic transforms (NTTs), and norm-based operations.

## Core Types

- **`Zq`** — Integer ring modulo `q`.
- **`Rq` / `Tq`** — Polynomial ring `Zq[X]/(Xⁿ + 1)`; `Tq` refers to the same type in the NTT (frequency) domain.

## Demonstrated APIs

- **Negacyclic NTT**  
  Forward and inverse Number-Theoretic Transforms over `Rq`.

- **Polynomial Ring Matrix Multiplication**  
  Matrix and vector multiplication over `Rq`, including Ajtai-style commitments and dot products.

- **Balanced Decomposition**  
  Decomposition of `Rq` elements into base-`b` digits using a balanced digit representation.

- **Norm Checking (`Zq`)**  
  Computation and verification of ℓ₂ and ℓ∞ norms over vectors in `Zq`.

- **Johnson–Lindenstrauss Projection (`Zq`)**  
  Projection of `Zq` vectors into lower dimensions using pseudorandom JL matrices.

- **Vector APIs for `Rq`**  
  Operations over vectors of polynomials, including aggregation and element-wise summation.

- **Matrix Transpose (`Rq`)**  
  Efficient transposition of matrices over `Rq`.

- **Random Sampling**  
  Seeded random generation of elements in `Zq` and `Rq`.

- **Challenge Space Sampling (`Rq`)**  
  Sampling of challenge polynomials with constrained coefficients (e.g., `{0, ±1, ±2}`).
