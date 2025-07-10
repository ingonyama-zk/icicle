# Matrix Operations (Rust bindings)

`icicle-core` exposes a set of **matrix primitives** that operate on data located either in host memory or on the GPU. These are implemented on top of – and share the same configuration structure as – the generic vector-operations backend (`VecOps`).

---
## Configuration: `MatMulConfig`

Matrix multiplication uses a dedicated configuration struct, `MatMulConfig`, which controls device placement, batching, transposition, and more:

```rust
use icicle_runtime::stream::IcicleStreamHandle;
use icicle_runtime::config::ConfigExtension;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MatMulConfig {
    pub stream_handle: IcicleStreamHandle, // Execution stream (e.g., CUDA stream)
    pub is_a_on_device: bool,              // True if `a` is on device memory
    pub is_b_on_device: bool,              // True if `b` is on device memory
    pub is_result_on_device: bool,         // True if result stays on device
    pub a_transposed: bool,                // Transpose input `a`
    pub b_transposed: bool,                // Transpose input `b`
    pub result_transposed: bool,           // Transpose the output
    pub is_async: bool,                    // Non-blocking execution if true
    pub ext: ConfigExtension,              // Backend-specific config
}

impl MatMulConfig {
    pub fn default() -> Self { /* ... */ }
}
```

- Use `MatMulConfig::default()` for standard single-matrix multiplication on the main device.
- For matrix transpose, use `VecOpsConfig` as before.

---
## Trait: `MatrixOps`

```rust
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_core::matrix_ops::MatMulConfig;
use icicle_core::vec_ops::VecOpsConfig;
use icicle_runtime::errors::IcicleError;

pub trait MatrixOps<T> {
    /// Performs matrix multiplication: `result = a × b`
    ///
    /// - `a`: shape `(a_rows × a_cols)` (row-major)
    /// - `b`: shape `(b_rows × b_cols)` (row-major)
    /// - `result`: shape `(a_rows × b_cols)` (row-major, must be preallocated)
    ///
    /// Requirements:
    /// - `a_cols == b_rows`
    /// - All buffers may reside in host or device memory
    fn matmul(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        a_rows: u32,
        a_cols: u32,
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        b_rows: u32,
        b_cols: u32,
        cfg: &MatMulConfig,
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), IcicleError>;

    /// Computes the transpose of a matrix in row-major order.
    ///
    /// - `input`: shape `(nof_rows × nof_cols)`
    /// - `output`: shape `(nof_cols × nof_rows)` (must be preallocated)
    ///
    /// Both input and output can reside on host or device memory.
    fn matrix_transpose(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        nof_rows: u32,
        nof_cols: u32,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), IcicleError>;
}
```

All concrete field / ring crates (for example `icicle_bn254`, `icicle_babybear`, …) re-export blanket implementations for their native scalar type via an internal macro.  Thus **you only need to import the scalar type** – the trait implementation is already in scope.

---
## Convenience free functions

Instead of calling the trait manually, you can use the thin wrappers defined in `icicle_core::matrix_ops`:

```rust
use icicle_core::matrix_ops::{matmul, matrix_transpose};
```

- `matmul` uses `MatMulConfig` for configuration.
- `matrix_transpose` uses `VecOpsConfig` for configuration.

---
## Example

Multiply two random BN254 matrices entirely on the GPU and read the result back to the host. (All buffers can be on host or device; you can mix and match as needed.)

```rust
use icicle_bn254::field::ScalarField;
use icicle_core::matrix_ops::{matmul, MatMulConfig};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_core::traits::GenerateRandom;

const N: usize = 512; // We will compute C = A × B where A,B are N×N

// 1. Generate random data on the host
let a_host = ScalarField::generate_random(N * N);
let b_host = ScalarField::generate_random(N * N);
// 2. Move the data to device memory
// 3. Allocate the result buffer on the device
// 4. Perform matmul
let cfg = MatMulConfig::default();
matmul(&a_dev[..], N as u32, N as u32,
       &b_dev[..], N as u32, N as u32,
       &cfg, &mut c_dev[..]).unwrap();
// Result is stored in c_dev for this example
// 5. Copy the result back if needed
```

---
## Error handling
All functions return `IcicleError`. The helpers perform validity checks (dimension mismatches, device/host placement, etc.) before dispatching to the backend, guaranteeing early and descriptive error messages. Checks include:
- Input and output buffer sizes must match the specified matrix dimensions.
- All buffers must be allocated on the correct device (if using device memory).
- For `matmul`, the inner dimensions must match (`a_cols == b_rows`).
- Output buffer must be preallocated to the correct size.

---
## Memory placement
- All buffers (`a`, `b`, `result`, `input`, `output`) can be on host or device memory.
- You can mix host and device buffers as needed; the API will handle transfers as required.
- Use `DeviceVec` for device memory and `HostSlice` for host memory.
- The `MatMulConfig` and `VecOpsConfig` structs control backend selection and options.

---
As of the current branch, there are **no batched matrix operations** exposed in the Rust bindings. Only `matmul` and `matrix_transpose` are available. 