# Icicle Example: Vector Operations API

## Key-Takeaway

The Vector Operations API supports the following:

 - element-wise vector operations (e.g. addition, multiplication)
 - vector reduction operations (e.g. sum of elements, product of elements)
 - scalar-vector operations (e.g add scalar to vector)
 - matrix operations (e.g. transposition)
 - miscellaneous operations like bit-reversal and slicing. 
 
 All these operations can be performed on a host or device both synchronously and asynchronously.

## Running the example

```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```

## What's in the example

1. `example_element_wise`: element-wise operations
2. `example_reduction`: reduction operations
3. `example_scalar_vector`: scalar-vector operations


