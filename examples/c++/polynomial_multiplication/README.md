# Icicle Example: Polynomial Multiplication with NTT

## Key-Takeaway

Icicle provides polynomial multiplication using the Number Theoretical Transform (NTT), including forward and inverse transforms.

## Concise Usage Explanation

1.	Include the necessary headers.
2.	Initialize the NTT domain.
3.	Prepare and transform the polynomials from host to device memory.
4.	Perform pointwise multiplication.
5.	Apply the inverse NTT.

## Running the example

- `cd` to your example directory
- compile with  `./compile.sh`
- run with `./run.sh`

## What's in the example

1.	Define the size of the example.
2.	Initialize input polynomials.
3.	Perform Radix-2 or Mixed-Radix NTT.
4.	Perform pointwise polynomial multiplication.
5.	Apply the inverse NTT.
