---
slug: /icicle/primitives/pairings
title: Pairings in ICICLE
---

# Pairings in ICICLE

Pairings are a fundamental cryptographic primitive that enable a wide range of advanced cryptographic protocols, including zero-knowledge proofs, identity-based encryption, and more. ICICLE provides efficient implementations of cryptographic pairings optimized for various hardware backends.

## What are Pairings?

A cryptographic pairing is a bilinear map e: G1 × G2 → GT, where:
- G1 and G2 are elliptic curve groups
- GT is a multiplicative subgroup of a field extension
- The map preserves the bilinear property: e(aP, bQ) = e(P,Q)^(ab)

This bilinear property makes pairings particularly useful for constructing complex cryptographic protocols.

## Pairing Implementation in ICICLE

ICICLE implements pairings through a templated interface that supports different pairing configurations. The main pairing function is defined in `pairing.h`:

```cpp
template <typename PairingConfig>
eIcicleError pairing(
  const typename PairingConfig::G1Affine& p,
  const typename PairingConfig::G2Affine& q,
  typename PairingConfig::TargetField* output);
```

### Key Components

1. **PairingConfig**: A configuration type that defines:
   - Field definitions
   - Implementation details
   - Group types (G1, G2)
   - Target field type (GT)

2. **Input Points**: The pairing takes two input points:
   - `p`: An affine point in G1
   - `q`: An affine point in G2

3. **Output**: The result is stored in the target field (GT)

## Supported Pairing Types

Currently, ICICLE supports the following pairing-friendly curves:
- bn254
- bls12-381
- bls12-377

The specific implementations can be found in the `models/` directory.

## Usage Example

Here's a basic example of how to use pairings in ICICLE:

```cpp
#include "icicle/pairing/pairing.h"
#include "icicle/pairing/models/bn254.h"

// Initialize points
Bn254::G1Affine p = ...;
Bn254::G2Affine q = ...;
Bn254::TargetField result;

// Compute pairing
eIcicleError err = icicle::pairing<Bn254>(p, q, &result);
```

## Further Reading

- [Architecture Overview](../arch_overview.md)
- [Getting Started Guide](../getting_started.md)
- [Programmer's Guide](../programmers_guide/general.md)

For specific implementation details and advanced usage, refer to the API documentation in the source code. 