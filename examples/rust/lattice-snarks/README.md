# Lattice SNARK Example — Types and APIs

This Rust example demonstrates core types and APIs used in lattice-based SNARK constructions. It is designed to serve as a reference and playground for key abstractions such as polynomial rings, NTTs, and norm-based operations.

## Types

1. **`Zq`** — Integer ring modulo `q`.
2. **`Rq/Tq`** — Polynomial ring `Zq[X]/(Xⁿ + 1)` (same type used for both `Tq`, the NTT domain representation).

## Demonstrated APIs

1. **Negacyclic NTT**  
   Forward and inverse Number-Theoretic Transforms for polynomials in `Rq`.

2. **Polynomial Ring Matrix Multiplication**  
   Matrix and vector multiplication over `Rq`, including Ajtai-commitment and dot products.

3. **Balanced Decomposition**  
   Decomposition of `Rq` elements into base-`b` digits using balanced representations.

4. **Norm Checking (`Zq`)**  
   Compute and verify ℓ₂ and ℓ-infinitiy norms of integer ring vectors.

5. **Johnson–Lindenstrauss Projection (`Zq`)**  
   Apply JL projections to `Zq` vectors using pseudorandom matrices

6. **Vector APIs for `Rq`**  
   Support for vectors of polynomials, including aggregation and vector-wise summation.

7. **Matrix Transpose (`Rq`)**  
   Efficient transposition of `Rq`-based matrices.

8. **Random Sampling**  
   Seeded random generation of `Zq` and `Rq` elements.

9. **Challenge Space Sampling (`Rq`)**  
   Sample challenge polynomials with constraints (e.g., `{0, ±1, ±2}`).

10. **Operator Norm Testing (`Rq`)**  
    Evaluate operator norm of `Rq` polynomials.

