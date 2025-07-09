# Matrix Operations (Rust bindings)

`icicle-core` exposes a small but very useful set of **matrix primitives** that work on data located either in host-memory or on the GPU.  They are implemented on top of – and share the same configuration structure as – the generic vector-operations backend (`VecOps`).

---
## Trait: `MatrixOps`

```rust
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_core::vec_ops::VecOpsConfig;
use icicle_runtime::errors::IcicleError;

pub trait MatrixOps<T> {
    /// Matrix multiplication:  `result = a × b` (row-major inputs / outputs)
    fn matmul(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        a_rows: u32,
        a_cols: u32,
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        b_rows: u32,
        b_cols: u32,
        cfg: &VecOpsConfig,
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), IcicleError>;

    /// Out-of-place matrix transpose (row-major)
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

Instead of calling the trait manually you can use the thin wrappers defined in `icicle_core::matrix_ops`:

```rust
use icicle_core::matrix_ops::{matmul, matrix_transpose};
```

They perform the exact same checks and dispatch to the trait implementation.

---
## Example

Multiply two random BN254 matrices entirely on the GPU and read the result back to the host.

```rust
use icicle_bn254::field::ScalarField;
use icicle_core::matrix_ops::{matmul};
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
let cfg = VecOpsConfig::default();
matmul(&a_dev[..], N as u32, N as u32,
       &b_dev[..], N as u32, N as u32,
       &cfg, &mut c_dev[..]).unwrap();
// Result is stored in c_dev for this example
// 5. Copy the result back if needed
```

---
## Error handling
All functions return `IcicleError`.  The helpers do validity checks (dimension mismatches, device/host placement, …) before dispatching to the backend, guaranteeing early and descriptive error messages. 