# ML-KEM (Kyber) – Golang bindings

:::note
For an in-depth explanation of the primitive, performance advice and backend-specific tuning knobs see the [C++ ML-KEM guide](../../cpp/lattice/pqc_ml_kem.md).
:::

## Overview

The `mlkem` package provides Golang bindings for ICICLE's batched **ML-KEM** implementation (Kyber). It exposes three main functions – **`Keygen`**, **`Encapsulate`** and **`Decapsulate`** – that work on host or device memory.

---

## Public API

### Key pair generation

```go
func Keygen(
    params KyberMode,
    entropy core.HostOrDeviceSlice,  // batch_size × 64 bytes
    config MlKemConfig,
    publicKeys core.HostOrDeviceSlice,  // batch_size × params.GetPublicKeyBytes()
    secretKeys core.HostOrDeviceSlice,  // batch_size × params.GetSecretKeyBytes()
) runtime.EIcicleError
```

### Encapsulation

```go
func Encapsulate(
    params KyberMode,
    message core.HostOrDeviceSlice,     // batch_size × 32 bytes
    publicKeys core.HostOrDeviceSlice,  // batch_size × params.GetPublicKeyBytes()
    config MlKemConfig,
    ciphertexts core.HostOrDeviceSlice,  // batch_size × params.GetCiphertextBytes()
    sharedSecrets core.HostOrDeviceSlice, // batch_size × params.GetSharedSecretBytes()
) runtime.EIcicleError
```

### Decapsulation

```go
func Decapsulate(
    params KyberMode,
    secretKeys core.HostOrDeviceSlice,    // batch_size × params.GetSecretKeyBytes()
    ciphertexts core.HostOrDeviceSlice,   // batch_size × params.GetCiphertextBytes()
    config MlKemConfig,
    sharedSecrets core.HostOrDeviceSlice, // batch_size × params.GetSharedSecretBytes()
) runtime.EIcicleError
```

All buffers may live either on the **host** or on the currently-active **device**.

### `MlKemConfig`

```go
type MlKemConfig struct {
    StreamHandle          runtime.Stream
    IsAsync               bool
    messagesOnDevice      bool
    entropyOnDevice       bool
    publicKeysOnDevice    bool
    secretKeysOnDevice    bool
    ciphertextsOnDevice   bool
    sharedSecretsOnDevice bool
    BatchSize             uint64
    Ext                   config_extension.ConfigExtensionHandler
}
```

Use `GetDefaultMlKemConfig()` to get the default configuration. Setting `IsAsync = true` lets the call return immediately; remember to synchronize the stream before reading results.

### Kyber Parameters

The package supports three parameter sets through the `KyberMode` enum:

```go
type KyberMode int

const (
    Kyber512  KyberMode = iota  // Security level 1 (128-bit security)
    Kyber768                    // Security level 3 (192-bit security)  
    Kyber1024                   // Security level 5 (256-bit security)
)
```

Each `KyberMode` provides methods to access parameter sizes:

```go
func (mode KyberMode) GetPublicKeyBytes() int    // Public key size in bytes
func (mode KyberMode) GetSecretKeyBytes() int    // Secret key size in bytes
func (mode KyberMode) GetCiphertextBytes() int   // Ciphertext size in bytes
func (mode KyberMode) GetSharedSecretBytes() int // Shared secret size in bytes (always 32)
```

Constants:

- `ENTROPY_BYTES = 64` - entropy required per keypair generation
- `MESSAGE_BYTES = 32` - message size for encapsulation

---

## Error handling

All functions return `runtime.EIcicleError`. An error is raised for invalid buffer sizes, mismatched device selection, or when the selected backend is not available. Check for `runtime.Success` to verify successful execution.

---

## See also

- [C++ ML-KEM guide](../../cpp/lattice/pqc_ml_kem.md)
- [Icicle runtime documentation](../../../start/architecture/multi-device.md) – streams, device management, buffer helpers
