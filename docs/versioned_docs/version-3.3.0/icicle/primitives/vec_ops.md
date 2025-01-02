
# Vector Operations API

## Overview

The Vector Operations API in Icicle provides a set of functions for performing element-wise and scalar-vector operations on vectors, matrix operations, and miscellaneous operations like bit-reversal and slicing. These operations can be performed on the host or device, with support for asynchronous execution.

### VecOpsConfig

The `VecOpsConfig` struct is a configuration object used to specify parameters for vector operations.

#### Fields

- **`stream: icicleStreamHandle`**: Specifies the CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
- **`is_a_on_device: bool`**: Indicates whether the first input vector (`a`) is already on the device. If `false`, the vector will be copied from the host to the device.
- **`is_b_on_device: bool`**: Indicates whether the second input vector (`b`) is already on the device. If `false`, the vector will be copied from the host to the device. This field is optional.
- **`is_result_on_device: bool`**: Indicates whether the result should be stored on the device. If `false`, the result will be transferred back to the host.
- **`is_async: bool`**: Specifies whether the vector operation should be performed asynchronously. When `true`, the operation will not block the CPU, allowing other operations to proceed concurrently. Asynchronous execution requires careful synchronization to ensure data integrity.
- **`batch_size: int`**: Number of vectors (or operations) to process in a batch. Each vector operation will be performed independently on each batch element.
- **`columns_batch: bool`**: True if the batched vectors are stored as columns in a 2D array (i.e., the vectors are strided in memory as columns of a matrix). If false, the batched vectors are stored contiguously in memory (e.g., as rows or in a flat array).
- **`ext: ConfigExtension*`**: Backend-specific extensions.

#### Default Configuration

```cpp
static VecOpsConfig default_vec_ops_config() {
    VecOpsConfig config = {
      nullptr, // stream
      false,   // is_a_on_device
      false,   // is_b_on_device
      false,   // is_result_on_device
      false,   // is_async
      1,       // batch_size
      false,   // columns_batch
      nullptr  // ext
    };
    return config;
}
```

### Element-wise Operations

These functions perform element-wise operations on two input vectors a and b. If VecOpsConfig specifies a batch_size greater than one, the operations are performed on multiple pairs of vectors simultaneously, producing corresponding output vectors.

#### `vector_add`

Adds two vectors element-wise.

```cpp
template <typename T>
eIcicleError vector_add(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);
```

#### `vector_sub`

Subtracts vector `b` from vector `a` element-wise.

```cpp
template <typename T>
eIcicleError vector_sub(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);
```

#### `vector_mul`

Multiplies two vectors element-wise.

```cpp
template <typename T>
eIcicleError vector_mul(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);
```

#### `vector_div`

Divides vector `a` by vector `b` element-wise.

```cpp
template <typename T>
eIcicleError vector_div(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);
```

#### `vector_accumulate`

Adds vector b to a, inplace.

```cpp
template <typename T>
eIcicleError vector_accumulate(T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config);
```

#### `convert_montogomery`

Convert a vector of field elements to/from montgomery form.
```cpp
template <typename T>
eIcicleError convert_montgomery(const T* input, uint64_t size, bool is_into, const VecOpsConfig& config, T* output);
```

### Reduction operations

These functions perform reduction operations on vectors. If VecOpsConfig specifies a batch_size greater than one, the operations are performed on multiple vectors simultaneously, producing corresponding output values. The storage arrangement of batched vectors is determined by the columns_batch field in the VecOpsConfig.

#### `vector_sum`

Computes the sum of all elements in each vector in a batch.

```cpp
template <typename T>
eIcicleError vector_sum(const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output);
```

#### `vector_product`

Computes the product of all elements in each vector in a batch.

```cpp
template <typename T>
eIcicleError vector_product(const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output);
```

### Scalar-Vector Operations

These functions apply a scalar operation to each element of a vector. If VecOpsConfig specifies a batch_size greater than one, the operations are performed on multiple vector-scalar pairs simultaneously, producing corresponding output vectors.

#### `scalar_add_vec / scalar_sub_vec`

Adds a scalar to each element of a vector.

```cpp
template <typename T>
eIcicleError scalar_add_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);
```

#### `scalar_sub_vec`

Subtract each element of a vector from a scalar `scalar-vec`.

```cpp
template <typename T>
eIcicleError scalar_sub_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);
```

#### `scalar_mul_vec`

Multiplies each element of a vector by a scalar.

```cpp
template <typename T>
eIcicleError scalar_mul_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);
```

### Matrix Operations

These functions perform operations on matrices. If VecOpsConfig specifies a batch_size greater than one, the operations are performed on multiple matrices simultaneously, producing corresponding output matrices.

#### `matrix_transpose`

Transposes a matrix.

```cpp
template <typename T>
eIcicleError matrix_transpose(const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out);
```

### Miscellaneous Operations

#### `bit_reverse`

Reorders the vector elements based on a bit-reversal pattern. If VecOpsConfig specifies a batch_size greater than one, the operation is performed on multiple vectors simultaneously.

```cpp
template <typename T>
eIcicleError bit_reverse(const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out);
```

#### `slice`

Extracts a slice from a vector. If VecOpsConfig specifies a batch_size greater than one, the operation is performed on multiple vectors simultaneously, producing corresponding output vectors.

```cpp
template <typename T>
eIcicleError slice(const T* vec_in, uint64_t offset, uint64_t stride, uint64_t size_in, uint64_t size_out, const VecOpsConfig& config, T* vec_out);
```

#### `highest_non_zero_idx`

Finds the highest non-zero index in a vector. If VecOpsConfig specifies a batch_size greater than one, the operation is performed on multiple vectors simultaneously.

```cpp
template <typename T>
eIcicleError highest_non_zero_idx(const T* vec_in, uint64_t size, const VecOpsConfig& config, int64_t* out_idx);
```

#### `polynomial_eval`

Evaluates a polynomial at given domain points. If VecOpsConfig specifies a batch_size greater than one, the operation is performed on multiple vectors simultaneously.

```cpp
template <typename T>
eIcicleError polynomial_eval(const T* coeffs, uint64_t coeffs_size, const T* domain, uint64_t domain_size, const VecOpsConfig& config, T* evals /*OUT*/);
```

#### `polynomial_division`

Divides two polynomials. If VecOpsConfig specifies a batch_size greater than one, the operation is performed on multiple vectors simultaneously.

```cpp
template <typename T>
eIcicleError polynomial_division(const T* numerator, int64_t numerator_deg, const T* denumerator, int64_t denumerator_deg, const VecOpsConfig& config, T* q_out /*OUT*/, uint64_t q_size, T* r_out /*OUT*/, uint64_t r_size);
```

### Rust and Go bindings

- [Golang](../golang-bindings/vec-ops.md)
- [Rust](../rust-bindings/vec-ops.md)
