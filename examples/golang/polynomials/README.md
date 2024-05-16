# ICICLE example: Polynomials in Golang

`ICICLE` provides Golang bindings to CUDA-accelerated C++ implementation of [Polynomials](https://dev.ingonyama.com/icicle/polynomials/overview).

## Usage
### Backend Initialization
```go
InitPolyBackend()
```
### Construction

```go
poly1 := CreateFromCoeffecitients(/* Coefficients of polynomial */ coeffs)
poly2 := CreateFromROUEvaluations(/* evaluations */ evals)
poly3 := Clone(/* polynomial to clone */ poly1)
```

### Arithmetic

```go
polyAdd := poly1.Add(&poly2)
polySub := poly1.Subtract(&poly2)
polyMul := poly1.Multiply(&poly2)
polyMulScalar := MultiplyByScalar(scalar)
quotient, remainder := poly1.Divide(&poly2)
```

### Evaluation

```go
ev := poly1.Eval(scalar)
ev2 := poly1.EvalOnDomain(scalars)
```

In this example we use `BN254` and `Babybear` fields. The examples shows arithmetic operations and evaluations execution.

## What's in the example

1. Define the size of polynomials. 
2. Initialize backends.
3. Generate random polynomials.
4. Execute arithmetic operations.
5. Execute evaluations.
6. Execute slicing.

Running the example:
```sh
go run main.go
```