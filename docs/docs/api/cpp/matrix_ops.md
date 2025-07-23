# Matrix Operations (C++ API)

ICICLE provides efficient, flexible matrix operations for cryptographic and scientific computing, supporting both CPU and GPU backends. The API is designed for high performance, batch processing, and seamless device/host memory management.

---
## Configuration: `MatMulConfig`

All matrix operations are controlled via the `MatMulConfig` struct, which allows you to specify device placement, batching, transposition, and more.

```cpp
struct MatMulConfig {
    icicleStreamHandle stream = nullptr; // stream for async execution.
    bool is_a_on_device = false;         // True if `a` resides on device memory.
    bool is_b_on_device = false;         // True if `b` resides on device memory.
    bool is_result_on_device = false;    // True to keep result on device, else host.
    bool a_transposed = false;           // True if `a` is pre-transposed.
    bool b_transposed = false;           // True if `b` is pre-transposed.
    bool result_transposed = false;      // True to transpose the output.
    bool is_async = false;               // True for non-blocking call; user must sync stream.
    ConfigExtension* ext = nullptr;      // Optional backend-specific settings.
  };
```

- Use `default_mat_mul_config()` to get a default config for standard (single, synchronous, host) operation.
- For most users, set `is_a_on_device`, `is_b_on_device`, and `is_result_on_device` to match your memory locations.
- Set `is_async = true` for non-blocking GPU execution (requires explicit synchronization).
- Use batching for high throughput on large workloads.

---
## Matrix Multiplication API

```cpp
template <typename T>
eIcicleError matmul(
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out);
```

- **Inputs:**
  - `mat_a`, `mat_b`: Pointers to input matrices (row-major order)
  - `nof_rows_a`, `nof_cols_a`: Dimensions of A
  - `nof_rows_b`, `nof_cols_b`: Dimensions of B
  - `config`: Matrix operation configuration (see above)
  - `mat_out`: Pointer to output matrix (row-major, must be preallocated)
- **Requirements:**
  - `nof_cols_a == nof_rows_b`
  - All pointers must be valid and point to memory on the correct device (host or GPU)
  - For batching, input/output pointers must be sized appropriately
- **Returns:** `eIcicleError` indicating success or failure
- **Notes:**
  - Supports both host and device memory (see config)
  - Supports batching and transposed inputs/outputs
  - Asynchronous execution is available via `is_async` and `stream`

---
## Example: Multiply Two Matrices on the GPU

```cpp
#include "icicle/mat_ops.h"
#include <vector>

using namespace icicle;

int main() {
    const uint32_t N = 512;
    std::vector<float> a(N * N), b(N * N), c(N * N);
    // ... fill a and b with data ...

    // Move data to device (if needed) using your preferred memory management
    // For this example, assume pointers a_dev, b_dev, c_dev are on device

    MatMulConfig cfg = default_mat_mul_config();
    cfg.is_a_on_device = true;
    cfg.is_b_on_device = true;
    cfg.is_result_on_device = true;

    eIcicleError err = matmul(
        a_dev, N, N,
        b_dev, N, N,
        cfg,
        c_dev
    );
    // ... check err, copy c_dev back to host if needed ...
}
```

---
## Notes
- All matrices are row-major by default; use the transposition flags for column-major or transposed operations.
- Batched operations are supported for high throughput.
- Both CPU and GPU (CUDA) backends are available out of the box.
- For more advanced usage (polynomial rings, custom types), see the full C++ API and backend headers.

---
## See Also
- [Vector Operations (vec_ops.md)](./vec_ops.md)
- [NTT, MSM, and other primitives](/api/overview.md)
- [Backend registration and extension (for advanced users)] 