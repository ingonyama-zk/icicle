# Field Implementations

This document describes the field implementations available in the ICICLE Golang bindings.

## Supported Fields

The following fields are supported:

- babybear

## Field Operations

Each field implementation provides the following operations:

### Basic Arithmetic

```go
// Field represents a field element
type Field struct {
    // ... internal fields
}

// Add adds two field elements
func (f *Field) Add(other *Field) (*Field, error)

// Sub subtracts two field elements
func (f *Field) Sub(other *Field) (*Field, error)

// Mul multiplies two field elements
func (f *Field) Mul(other *Field) (*Field, error)

// Div divides two field elements
func (f *Field) Div(other *Field) (*Field, error)

// Neg negates a field element
func (f *Field) Neg() (*Field, error)

// Inv computes the multiplicative inverse
func (f *Field) Inv() (*Field, error)

// Pow raises a field element to a power
func (f *Field) Pow(exp uint64) (*Field, error)
```

### Vector Operations

```go
// VecOps provides vector operations for field elements
type VecOps struct {
    // ... internal fields
}

// Add performs element-wise addition
func (v *VecOps) Add(a, b []Field) ([]Field, error)

// Sub performs element-wise subtraction
func (v *VecOps) Sub(a, b []Field) ([]Field, error)

// Mul performs element-wise multiplication
func (v *VecOps) Mul(a, b []Field) ([]Field, error)

// Neg performs element-wise negation
func (v *VecOps) Neg(a []Field) ([]Field, error)
```

### NTT Operations

```go
// NTT provides Number Theoretic Transform operations
type NTT struct {
    // ... internal fields
}

// Forward performs forward NTT
func (n *NTT) Forward(a []Field) ([]Field, error)

// Inverse performs inverse NTT
func (n *NTT) Inverse(a []Field) ([]Field, error)

// BatchForward performs batch forward NTT
func (n *NTT) BatchForward(a []Field, batchSize int) ([]Field, error)

// BatchInverse performs batch inverse NTT
func (n *NTT) BatchInverse(a []Field, batchSize int) ([]Field, error)
```

### Extension Field Operations

```go
// ExtensionField represents an extension field element
type ExtensionField struct {
    // ... internal fields
}

// Add adds two extension field elements
func (f *ExtensionField) Add(other *ExtensionField) (*ExtensionField, error)

// Sub subtracts two extension field elements
func (f *ExtensionField) Sub(other *ExtensionField) (*ExtensionField, error)

// Mul multiplies two extension field elements
func (f *ExtensionField) Mul(other *ExtensionField) (*ExtensionField, error)

// Div divides two extension field elements
func (f *ExtensionField) Div(other *ExtensionField) (*ExtensionField, error)
```

## Usage Example

Here's an example of how to use field operations:

```go
package main

import (
    "fmt"
    "icicle/fields/babybear"
)

func main() {
    // Create field elements
    a := babybear.NewField(1)
    b := babybear.NewField(2)

    // Perform arithmetic
    sum, err := a.Add(b)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Sum: %v\n", sum)
}
```

## Supported Operations by Field

| Operation\Field | babybear |
| --- | :---: |
| Basic Arithmetic | ✅ |
| Vector Operations | ✅ |
| NTT | ✅ |
| Extension Field | ✅ |

## Error Handling

All field operations return an error that should be checked:

```go
sum, err := a.Add(b)
if err != nil {
    // Handle error
}
```

Common errors include:
- Division by zero
- Invalid field element values
- Memory allocation failures
- Device errors

## Performance Considerations

1. Use vector operations for batch processing
2. Use NTT for polynomial multiplication
3. Use Montgomery form for better performance
4. Consider using extension fields only when necessary
5. Use batch operations to amortize overhead

## Memory Management

When working with field elements:
1. Use vector operations for better performance
2. Keep elements in Montgomery form when possible
3. Use batch operations to reduce memory transfers
4. Be mindful of device memory limitations
5. Free memory when no longer needed 