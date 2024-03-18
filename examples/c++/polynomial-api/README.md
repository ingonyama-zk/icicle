# ICICLE examples: computations with polynomials

## Best-Practices

We recommend to run our examples in [ZK-containers](../../ZK-containers.md) to save your time and mental energy.

## Key-Takeaway

Polynomials are crucial for Zero-Knowledge Proofs (ZKPs): they enable efficient representation and verification of computational statements, facilitate privacy-preserving protocols, and support complex mathematical operations essential for constructing and verifying proofs without revealing underlying data.

## Running the example

Make sure you have compiled a curve-specific `ICICLE` library.
For example, for bn254 curve execute the following from the project root directory:

```sh
cd icicle
mkdir -p build
cmake -DBUILD_TESTS=ON -DG2_DEFINED=ON -DCURVE=bn254 -S . -B build
cmake --build build
```

To run example, from project root directory:

```sh
cd examples/c++/polynomial-api
./compile.sh
./run.sh
```

To change the curve, edit `CURVE_ID` in `CMakeLists.txt`

## What's in the examples

- `example_evaluate`: Make polynomial from coefficients and evalue it at random point.

- `example_clone`: Make a separate copy of a polynomial.

- `example_from_rou`: Reconstruct polynomial from values at the roots of unity. This operation is a cornerstone in the efficient implementation of zero-knowledge proofs, particularly in the areas of proof construction, verification, and polynomial arithmetic. By leveraging the algebraic structure and computational properties of roots of unity, ZKP protocols can achieve the scalability, efficiency, and privacy necessary for practical applications in blockchain, secure computation, and beyond.

- `example_addition`, `example_addition_inplace`: Different flavors of polynomial addition.

- `example_multiplication`: A product of two polynimials

- `example_multiplicationScalar`: A product of scalar and a polynomial.

- `example_monomials`: Add/subtract a monomial to a polynom. Monomial is a single term, which is the product of a constant coefficient and a variable raised to a non-negative integer power.

- `example_ReadCoeffsToHost`: Download coefficients of a polynomial to a host. `ICICLE` keeps all polynomials on GPU, for on-host operation one needs such an operation.

- `example_divisionSmall`, `example_divisionLarge`: Different flavors of division.

- `example_divideByVanishingPolynomial`: A vanishing polynomial over a set S is a polynomial that evaluates to zero for every element in S. For a simple case, consider the set S={a}, a single element. The polynomial f(x)=x−a vanishes over S because f(a)=0. Mathematically, dividing a polynomial P(x) by a vanishing polynomial V(x) typically involves finding another polynomial Q(x) and possibly a remainder R(x) such that P(x)=Q(x)V(x)+R(x), where R(x) has a lower degree than V(x). In many cryptographic applications, the focus is on ensuring that P(x) is exactly divisible by V(x), meaning R(x)=0.
