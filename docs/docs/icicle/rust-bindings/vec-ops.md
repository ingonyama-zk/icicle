# Vector Operations API

Our vector operations API which is part of `icicle-cuda-runtime` package, includes fundamental methods for addition, subtraction, and multiplication of vectors, with support for both host and device memory. 


## Supported curves

Vector operations are supported on the following curves:

`bls12-377`, `bls12-381`, `bn-254`, `bw6-761`, `grumpkin`

## Examples

### Addition of Scalars

```rust
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::vec_ops::{add_scalars};


let test_size = 1 << 18;

let a: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let b: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let mut result: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);

let cfg = VecOpsConfig::default();
add_scalars(&a, &b, &mut result, &cfg).unwrap();
```

### Subtraction of Scalars

```rust
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::vec_ops::{sub_scalars};

let test_size = 1 << 18;

let a: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let b: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let mut result: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);

let cfg = VecOpsConfig::default();
sub_scalars(&a, &b, &mut result, &cfg).unwrap();
```

### Multiplication of Scalars

```rust
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::vec_ops::{mul_scalars};

let a: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
let ones: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::one(); test_size]);
let mut result: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);

let cfg = VecOpsConfig::default();
mul_scalars(&a, &ones, &mut result, &cfg).unwrap();
```


## Vector Operations Configuration

The `VecOpsConfig` struct encapsulates the settings for vector operations, including device context and operation modes.

### `VecOpsConfig`

Defines configuration parameters for vector operations.

```rust
pub struct VecOpsConfig<'a> {
    pub ctx: DeviceContext<'a>,
    is_a_on_device: bool,
    is_b_on_device: bool,
    is_result_on_device: bool,
    is_result_montgomery_form: bool,
    pub is_async: bool,
}
```

#### Fields

- **`ctx: DeviceContext<'a>`**: Specifies the device context for the operation, including the device ID and memory pool.
- **`is_a_on_device`**: Indicates if the first operand vector resides in device memory.
- **`is_b_on_device`**: Indicates if the second operand vector resides in device memory.
- **`is_result_on_device`**: Specifies if the result vector should be stored in device memory.
- **`is_result_montgomery_form`**: Determines if the result should be in Montgomery form.
- **`is_async`**: Enables asynchronous operation. If `true`, operations are non-blocking; otherwise, they block the current thread.

### Default Configuration

`VecOpsConfig` can be initialized with default settings tailored for a specific device:

```
let cfg = VecOpsConfig::default();
```

These are the default settings.

```rust
impl<'a> Default for VecOpsConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> VecOpsConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        VecOpsConfig {
            ctx: DeviceContext::default_for_device(device_id),
            is_a_on_device: false,
            is_b_on_device: false,
            is_result_on_device: false,
            is_result_montgomery_form: false,
            is_async: false,
        }
    }
}
```

## Vector Operations

Vector operations are implemented through the `VecOps` trait, these traits are implemented for all [supported curves](#supported-curves) providing methods for addition, subtraction, and multiplication of vectors.

### `VecOps` Trait

```rust
pub trait VecOps<F> {
    fn add(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn sub(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn mul(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;
}
```

#### Methods

- **`add`**: Computes the element-wise sum of two vectors.
- **`sub`**: Computes the element-wise difference between two vectors.
- **`mul`**: Performs element-wise multiplication of two vectors.

### Argument Validation

Before invoking any of the above vector operations, we always call `check_vec_ops_args`, to make sure that inputs `a` and `b` can be operated on with and that the results pointer can contain the result:

```rust
fn check_vec_ops_args<F>(a: &HostOrDeviceSlice<F>, b: &HostOrDeviceSlice<F>, result: &mut HostOrDeviceSlice<F>) {
    if a.len() != b.len() || a.len() != result.len() {
        panic!(
            "left, right and output lengths {}; {}; {} do not match",
            a.len(),
            b.len(),
            result.len()
        );
    }
}
```
