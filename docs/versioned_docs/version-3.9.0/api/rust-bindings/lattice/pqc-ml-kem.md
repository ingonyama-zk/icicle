# ML-KEM (Kyber) – Rust bindings

:::note
For an in-depth explanation of the primitive, performance advice and backend-specific tuning knobs see the [C++ ML-KEM guide](../cpp/pqc_ml_kem.md).
:::

## Overview

`icicle-ml-kem` is a Rust wrapper around Icicle's batched **ML-KEM** implementation (Kyber).  It exposes three functions – **`keygen`**, **`encapsulate`** and **`decapsulate`** – that work on host or device memory.

---

## Public API

### Key pair generation

```rust
pub fn keygen<P: KyberParams>(
    entropy: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 64 bytes
    config: &MlKemConfig,
    public_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
    secret_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
) -> Result<(), eIcicleError>;
```

### Encapsulation

```rust
pub fn encapsulate<P: KyberParams>(
    message: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 32 bytes
    public_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
    config: &MlKemConfig,
    ciphertexts: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
    shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
) -> Result<(), eIcicleError>;
```

### Decapsulation

```rust
pub fn decapsulate<P: KyberParams>(
    secret_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
    ciphertexts: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
    config: &MlKemConfig,
    shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
) -> Result<(), eIcicleError>;
```

All buffers may live either on the **host** or on the currently-active **device**.

### `MlKemConfig`

```rust
pub struct MlKemConfig {
    pub stream:                    IcicleStreamHandle,
    pub is_async:                  bool,

    // Location hints
    pub messages_on_device:        bool,
    pub entropy_on_device:         bool,
    pub public_keys_on_device:     bool,
    pub secret_keys_on_device:     bool,
    pub ciphertexts_on_device:     bool,
    pub shared_secrets_on_device:  bool,

    pub batch_size:                i32,
    pub ext:                       ConfigExtension,
}
```

use `Default::default()` to get the default configuration. Setting `is_async = true` lets the call return immediately; remember to synchronize the stream before reading results.

---

## Quick-start example (Kyber768, host buffers)

```rust
use icicle_ml_kem as mlkem;
use icicle_ml_kem::{keygen, encapsulate, decapsulate};
use icicle_ml_kem::kyber_params::{Kyber768Params, ENTROPY_BYTES, MESSAGE_BYTES};
use icicle_runtime::memory::HostSlice;
use rand::{RngCore, rngs::OsRng};

const BATCH: usize = 1 << 12;

fn main() {
    // Allocate buffers on the host
    let mut entropy         = vec![0u8; BATCH * ENTROPY_BYTES];
    let mut msg             = vec![0u8; BATCH * MESSAGE_BYTES];
    OsRng.fill_bytes(&mut entropy);
    OsRng.fill_bytes(&mut msg);

    let mut pk = vec![0u8; BATCH * Kyber768Params::PUBLIC_KEY_BYTES];
    let mut sk = vec![0u8; BATCH * Kyber768Params::SECRET_KEY_BYTES];
    let mut ct = vec![0u8; BATCH * Kyber768Params::CIPHERTEXT_BYTES];
    let mut ss_enc = vec![0u8; BATCH * Kyber768Params::SHARED_SECRET_BYTES];
    let mut ss_dec = vec![0u8; BATCH * Kyber768Params::SHARED_SECRET_BYTES];

    // Configuration – everything stays on host
    let mut cfg = mlkem::config::MlKemConfig::default();
    cfg.batch_size = BATCH as i32;

    // Key generation
    keygen::<Kyber768Params>(
        HostSlice::from_slice(&entropy),
        &cfg,
        HostSlice::from_mut_slice(&mut pk),
        HostSlice::from_mut_slice(&mut sk),
    ).unwrap();

    // Encapsulation
    encapsulate::<Kyber768Params>(
        HostSlice::from_slice(&msg),
        HostSlice::from_slice(&pk),
        &cfg,
        HostSlice::from_mut_slice(&mut ct),
        HostSlice::from_mut_slice(&mut ss_enc),
    ).unwrap();

    // Decapsulation
    decapsulate::<Kyber768Params>(
        HostSlice::from_slice(&sk),
        HostSlice::from_slice(&ct),
        &cfg,
        HostSlice::from_mut_slice(&mut ss_dec),
    ).unwrap();

    assert_eq!(ss_enc, ss_dec);
    println!("{} successful KEM operations!", BATCH);
}
```

---

## Device & async execution

The API works identically for Device buffers – allocate input/output `DeviceVec`s, and provide an `IcicleStream` for non-blocking execution. See [`tests.rs`](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-pqc/icicle-ml-kem/src/tests.rs) in the crate for an end-to-end example.

---

## Error handling

All functions return `Result<(), eIcicleError>`.  An error is raised for invalid buffer sizes, mismatched device selection, or when the selected backend is not available.

---

## See also

* [C++ ML-KEM guide](../cpp/lattice/pqc_ml_kem.md)
* [Icicle runtime documentation](multi-gpu.md) – streams, device management, buffer helpers
