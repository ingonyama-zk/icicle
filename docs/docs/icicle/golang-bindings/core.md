# Core Golang Functionality

This document describes the core functionality available in the ICICLE Golang bindings.

## Memory Management

### Slice Operations

The `slice.go` file provides memory management functionality for working with device memory:

```go
// DeviceSlice represents a slice of data allocated on the device
type DeviceSlice[T any] struct {
    // ... internal fields
}

// NewDeviceSlice creates a new device slice with the specified size
func NewDeviceSlice[T any](size int) (*DeviceSlice[T], error)

// CopyToDevice copies data from host to device
func (s *DeviceSlice[T]) CopyToDevice(data []T) error

// CopyFromDevice copies data from device to host
func (s *DeviceSlice[T]) CopyFromDevice() ([]T, error)

// Free releases the device memory
func (s *DeviceSlice[T]) Free() error
```

## Vector Operations

The `vec_ops.go` file provides vector operations for field elements:

```go
// VecOps provides vector operations for field elements
type VecOps[T any] struct {
    // ... internal fields
}

// Add performs element-wise addition
func (v *VecOps[T]) Add(a, b []T) ([]T, error)

// Sub performs element-wise subtraction
func (v *VecOps[T]) Sub(a, b []T) ([]T, error)

// Mul performs element-wise multiplication
func (v *VecOps[T]) Mul(a, b []T) ([]T, error)

// Neg performs element-wise negation
func (v *VecOps[T]) Neg(a []T) ([]T, error)
```

## Multi-Scalar Multiplication

The `msm.go` file provides multi-scalar multiplication operations:

```go
// MSM performs multi-scalar multiplication
type MSM[T any] struct {
    // ... internal fields
}

// Compute performs the multi-scalar multiplication
func (m *MSM[T]) Compute(points []T, scalars []T) (T, error)

// BatchCompute performs batch multi-scalar multiplication
func (m *MSM[T]) BatchCompute(points []T, scalars []T, batchSize int) ([]T, error)
```

## Number Theoretic Transform

The `ntt.go` file provides NTT operations:

```go
// NTT provides Number Theoretic Transform operations
type NTT[T any] struct {
    // ... internal fields
}

// Forward performs forward NTT
func (n *NTT[T]) Forward(a []T) ([]T, error)

// Inverse performs inverse NTT
func (n *NTT[T]) Inverse(a []T) ([]T, error)

// BatchForward performs batch forward NTT
func (n *NTT[T]) BatchForward(a []T, batchSize int) ([]T, error)

// BatchInverse performs batch inverse NTT
func (n *NTT[T]) BatchInverse(a []T, batchSize int) ([]T, error)
```

## Hash Functions

The `hash.go` file provides hash function operations:

```go
// Hash provides hash function operations
type Hash[T any] struct {
    // ... internal fields
}

// Poseidon performs Poseidon hash
func (h *Hash[T]) Poseidon(inputs []T) (T, error)

// Poseidon2 performs Poseidon2 hash
func (h *Hash[T]) Poseidon2(inputs []T) (T, error)

// BatchPoseidon performs batch Poseidon hash
func (h *Hash[T]) BatchPoseidon(inputs []T, batchSize int) ([]T, error)

// BatchPoseidon2 performs batch Poseidon2 hash
func (h *Hash[T]) BatchPoseidon2(inputs []T, batchSize int) ([]T, error)
```

## Merkle Tree

The `merkletree.go` file provides Merkle tree operations:

```go
// MerkleTree provides Merkle tree operations
type MerkleTree[T any] struct {
    // ... internal fields
}

// Build constructs a Merkle tree from leaves
func (m *MerkleTree[T]) Build(leaves []T) error

// GetRoot returns the root of the Merkle tree
func (m *MerkleTree[T]) GetRoot() (T, error)

// GetProof generates a Merkle proof for a leaf
func (m *MerkleTree[T]) GetProof(index int) ([]T, error)

// Verify verifies a Merkle proof
func (m *MerkleTree[T]) Verify(leaf T, proof []T, index int) (bool, error)
```

## Utility Functions

The `utils.go` file provides utility functions:

```go
// GetDeviceCount returns the number of available CUDA devices
func GetDeviceCount() (int, error)

// SetDevice sets the current CUDA device
func SetDevice(deviceID int) error

// GetDeviceMemoryInfo returns memory information for the current device
func GetDeviceMemoryInfo() (free, total uint64, err error)

// SyncDevice synchronizes the current device
func SyncDevice() error
```

## Usage Example

Here's an example of how to use the core functionality:

```go
package main

import (
    "fmt"
    "icicle/core"
)

func main() {
    // Initialize device
    if err := core.SetDevice(0); err != nil {
        panic(err)
    }

    // Create vectors
    a := []Field{1, 2, 3, 4}
    b := []Field{5, 6, 7, 8}

    // Perform vector operations
    vecOps := core.NewVecOps[Field]()
    result, err := vecOps.Add(a, b)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Result: %v\n", result)
}
```

## Supported Operations

The following table shows which operations are supported for each curve:

| Operation\Curve | bn254 | bls12_377 | bls12_381 | bw6-761 | grumpkin |
| --- | :---: | :---: | :---: | :---: | :---: |
| MSM | ✅ | ✅ | ✅ | ✅ | ✅ |
| G2  | ✅ | ✅ | ✅ | ✅ | ❌ |
| NTT | ✅ | ✅ | ✅ | ✅ | ❌ |
| ECNTT | ✅ | ✅ | ✅ | ✅ | ❌ |
| VecOps | ✅ | ✅ | ✅ | ✅ | ✅ |
| Polynomials | ✅ | ✅ | ✅ | ✅ | ❌ |

## Error Handling

All functions return an error that should be checked:

```go
result, err := vecOps.Add(a, b)
if err != nil {
    // Handle error
}
```

Common errors include:
- Device memory allocation failures
- Invalid input sizes
- Device synchronization errors
- CUDA runtime errors

## Memory Management

When working with device memory:
1. Always call `Free()` on device slices when they are no longer needed
2. Use `CopyToDevice()` and `CopyFromDevice()` to transfer data between host and device
3. Be mindful of device memory limitations
4. Use batch operations when possible to amortize memory transfer costs 