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
    params *KyberParams,
    entropy core.HostOrDeviceSlice,  // batch_size × 64 bytes
    config MlKemConfig,
    publicKeys core.HostOrDeviceSlice,  // batch_size × params.PublicKeyBytes
    secretKeys core.HostOrDeviceSlice,  // batch_size × params.SecretKeyBytes
) runtime.EIcicleError
```

### Encapsulation

```go
func Encapsulate(
    params *KyberParams,
    message core.HostOrDeviceSlice,     // batch_size × 32 bytes
    publicKeys core.HostOrDeviceSlice,  // batch_size × params.PublicKeyBytes
    config MlKemConfig,
    ciphertexts core.HostOrDeviceSlice,  // batch_size × params.CiphertextBytes
    sharedSecrets core.HostOrDeviceSlice, // batch_size × params.SharedSecretBytes
) runtime.EIcicleError
```

### Decapsulation

```go
func Decapsulate(
    params *KyberParams,
    secretKeys core.HostOrDeviceSlice,    // batch_size × params.SecretKeyBytes
    ciphertexts core.HostOrDeviceSlice,   // batch_size × params.CiphertextBytes
    config MlKemConfig,
    sharedSecrets core.HostOrDeviceSlice, // batch_size × params.SharedSecretBytes
) runtime.EIcicleError
```

All buffers may live either on the **host** or on the currently-active **device**.

### `MlKemConfig`

```go
type MlKemConfig struct {
    StreamHandle          runtime.Stream
    IsAsync               bool
    MessagesOnDevice      bool
    EntropyOnDevice       bool
    PublicKeysOnDevice    bool
    SecretKeysOnDevice    bool
    CiphertextsOnDevice   bool
    SharedSecretsOnDevice bool
    BatchSize             uint64
    Ext                   config_extension.ConfigExtensionHandler
}
```

Use `GetDefaultMlKemConfig()` to get the default configuration. Setting `IsAsync = true` lets the call return immediately; remember to synchronize the stream before reading results.

### Kyber Parameters

The package supports three parameter sets:

- **`Kyber512Params`**: Security level 1 (128-bit security)
- **`Kyber768Params`**: Security level 3 (192-bit security)
- **`Kyber1024Params`**: Security level 5 (256-bit security)

```go
type KyberParams struct {
    PublicKeyBytes    int
    SecretKeyBytes    int
    CiphertextBytes   int
    SharedSecretBytes int
    K                 uint8
    Eta1              uint8
    Eta2              uint8
    Du                uint8
    Dv                uint8
    // ... internal FFI functions
}
```

Constants:

- `ENTROPY_BYTES = 64` - entropy required per keypair generation
- `MESSAGE_BYTES = 32` - message size for encapsulation

---

## Quick-start example (Kyber768, host buffers)

```go
package main

import (
    "crypto/rand"
    "fmt"
    "log"

    "github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
    mlkem "github.com/ingonyama-zk/icicle/v3/wrappers/golang/pqc/ml-kem"
    "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

const BATCH = 1 << 12

func main() {
    runtime.LoadBackendFromEnvOrDefault()

    // Allocate buffers on the host
    params := &mlkem.Kyber768Params
    entropyLen := BATCH * mlkem.ENTROPY_BYTES
    messagesLen := BATCH * mlkem.MESSAGE_BYTES
    publicKeysLen := BATCH * params.PublicKeyBytes
    secretKeysLen := BATCH * params.SecretKeyBytes
    ciphertextsLen := BATCH * params.CiphertextBytes
    sharedSecretsLen := BATCH * params.SharedSecretBytes

    entropy := make([]byte, entropyLen)
    messages := make([]byte, messagesLen)
    publicKeys := make([]byte, publicKeysLen)
    secretKeys := make([]byte, secretKeysLen)
    ciphertexts := make([]byte, ciphertextsLen)
    sharedSecretsEnc := make([]byte, sharedSecretsLen)
    sharedSecretsDec := make([]byte, sharedSecretsLen)

    // Fill entropy and messages with randomness
    if _, err := rand.Read(entropy); err != nil {
        log.Fatalf("Failed to generate entropy: %v", err)
    }
    if _, err := rand.Read(messages); err != nil {
        log.Fatalf("Failed to generate messages: %v", err)
    }

    // Configuration – everything stays on host
    config := mlkem.GetDefaultMlKemConfig()
    config.BatchSize = BATCH

    // Initialize device
    device := runtime.CreateDevice("CUDA-PQC", 0)
    err := runtime.SetDevice(&device)
    if err != runtime.Success {
        log.Fatalf("Failed to set device: %v", err)
    }

    // Key generation
    err = mlkem.Keygen(params,
        core.HostSliceFromElements(entropy),
        config,
        core.HostSliceFromElements(publicKeys),
        core.HostSliceFromElements(secretKeys),
    )
    if err != runtime.Success {
        log.Fatalf("Keygen failed: %v", err)
    }

    // Encapsulation
    err = mlkem.Encapsulate(params,
        core.HostSliceFromElements(messages),
        core.HostSliceFromElements(publicKeys),
        config,
        core.HostSliceFromElements(ciphertexts),
        core.HostSliceFromElements(sharedSecretsEnc),
    )
    if err != runtime.Success {
        log.Fatalf("Encapsulate failed: %v", err)
    }

    // Decapsulation
    err = mlkem.Decapsulate(params,
        core.HostSliceFromElements(secretKeys),
        core.HostSliceFromElements(ciphertexts),
        config,
        core.HostSliceFromElements(sharedSecretsDec),
    )
    if err != runtime.Success {
        log.Fatalf("Decapsulate failed: %v", err)
    }

    // Verify shared secrets match
    for i := 0; i < len(sharedSecretsEnc); i++ {
        if sharedSecretsEnc[i] != sharedSecretsDec[i] {
            log.Fatalf("Shared secrets do not match at index %d", i)
        }
    }

    fmt.Printf("%d successful KEM operations!\n", BATCH)
}
```

---

## Device & async execution

The API works identically for Device buffers – allocate input/output `DeviceSlice`s, and provide a `Stream` for non-blocking execution:

```go
// Create device and stream
device := runtime.CreateDevice("CUDA-PQC", 0)
runtime.SetDevice(&device)

stream, err := runtime.CreateStream()
if err != runtime.Success {
    log.Fatalf("CreateStream failed: %v", err)
}

// Configure for async device execution
config := mlkem.GetDefaultMlKemConfig()
config.BatchSize = BATCH
config.IsAsync = true
config.StreamHandle = stream

// Allocate device buffers
var dEntropy, dMessages, dPublicKeys, dSecretKeys core.DeviceSlice
var dCiphertexts, dSharedSecretsEnc, dSharedSecretsDec core.DeviceSlice

// Copy host data to device
core.HostSliceFromElements(hostEntropy).CopyToDeviceAsync(&dEntropy, stream, true)
core.HostSliceFromElements(hostMessages).CopyToDeviceAsync(&dMessages, stream, true)

// Allocate device output buffers
dPublicKeys.MallocAsync(1, publicKeysLen, stream)
dSecretKeys.MallocAsync(1, secretKeysLen, stream)
dCiphertexts.MallocAsync(1, ciphertextsLen, stream)
dSharedSecretsEnc.MallocAsync(1, sharedSecretsLen, stream)
dSharedSecretsDec.MallocAsync(1, sharedSecretsLen, stream)

// Execute operations on device
mlkem.Keygen(params, dEntropy, config, dPublicKeys, dSecretKeys)
mlkem.Encapsulate(params, dMessages, dPublicKeys, config, dCiphertexts, dSharedSecretsEnc)
mlkem.Decapsulate(params, dSecretKeys, dCiphertexts, config, dSharedSecretsDec)

// Copy results back to host
core.HostSliceFromElements(hostResultsEnc).CopyFromDeviceAsync(&dSharedSecretsEnc, stream)
core.HostSliceFromElements(hostResultsDec).CopyFromDeviceAsync(&dSharedSecretsDec, stream)

// Synchronize and cleanup
runtime.SynchronizeStream(stream)
runtime.DestroyStream(stream)
```

---

## Error handling

All functions return `runtime.EIcicleError`. An error is raised for invalid buffer sizes, mismatched device selection, or when the selected backend is not available. Check for `runtime.Success` to verify successful execution.

---

## See also

- [C++ ML-KEM guide](../../cpp/lattice/pqc_ml_kem.md)
- [Icicle runtime documentation](../../../start/architecture/multi-device.md) – streams, device management, buffer helpers
