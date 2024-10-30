# ICICLE Hashing in Golang

:::note

For a general overview of ICICLE's hashing logic and supported algorithms, check out the [ICICLE Hashing Overview](../primitives/hash.md).

:::

:::caution Warning

Using the Hash package requires `go` version 1.22

:::

## Overview

The ICICLE library provides Golang bindings for hashing using a variety of cryptographic hash functions. These hash functions are optimized for both general-purpose data and cryptographic operations such as multi-scalar multiplication, commitment generation, and Merkle tree construction.

This guide will show you how to use the ICICLE hashing API in Golang with examples for common hash algorithms, such as Keccak-256, Keccak-512, SHA3-256, SHA3-512, and Blake2s.

## Importing Hash Functions

To use the hashing functions in Golang, you only need to import the hash package from the ICICLE Golang bindings. For example:

```go
import (
  "github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
)
```

## API Usage

### 1. Creating a Hasher Instance

Each hash algorithm can be instantiated by calling its respective constructor. The `New<Hash>Hasher` function takes an optional default input size, which can be set to 0 unless required for a specific use case.

Example for Keccak-256:

```go
keccakHasher := hash.NewKeccak256Hasher(0 /* default input size */)
```

### 2. Hashing a Simple String

Once you have created a hasher instance, you can hash any input data, such as strings or byte arrays, and store the result in an output buffer.
Here’s how to hash a simple string using Keccak-256:

```go
import (
  "encoding/hex"

  "github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
)

inputStrAsBytes := []bytes("I like ICICLE! It's so fast and simple")
keccakHasher, error := hash.NewKeccak256Hasher(0 /*default chunk size*/)
if error != runtime.Success {
  fmt.Println("error:", error)
  return
}

outputRef := make([]byte, 32)
keccakHasher.Hash(
  core.HostSliceFromElements(inputStrAsBytes),
  core.HostSliceFromElements(outputRef),
  core.GetDefaultHashConfig(),
)

// convert the output to a hex string for easy readability
outputAsHexStr = hex.EncodeToString(outputRef)
fmt.Println!("Hash(`", input_str, "`) =", &outputAsHexStr)
```

Using other hash algorithms is similar and only requires replacing the Hasher constructor with the relevant hashing algorithm.
