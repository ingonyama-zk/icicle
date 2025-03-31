# Curve Implementations

This document describes the curve implementations available in the ICICLE Golang bindings.

## Supported Curves

The following curves are supported:

- bn254
- bls12_377
- bls12_381
- bw6-761
- grumpkin

## Curve Operations

Each curve implementation provides the following operations:

### Point Operations

```go
// Point represents a point on the curve
type Point struct {
    X, Y Field
}

// Add adds two points
func (p *Point) Add(other *Point) (*Point, error)

// Mul multiplies a point by a scalar
func (p *Point) Mul(scalar Field) (*Point, error)

// Neg negates a point
func (p *Point) Neg() (*Point, error)

// IsOnCurve checks if a point is on the curve
func (p *Point) IsOnCurve() bool
```

### G2 Operations

G2 operations are available for all curves except grumpkin:

```go
// G2Point represents a point on the G2 curve
type G2Point struct {
    X, Y Field
}

// Add adds two G2 points
func (p *G2Point) Add(other *G2Point) (*G2Point, error)

// Mul multiplies a G2 point by a scalar
func (p *G2Point) Mul(scalar Field) (*G2Point, error)

// Pairing performs the pairing operation
func Pairing(p1 *Point, p2 *G2Point) (Field, error)
```

### ECNTT Operations

ECNTT operations are available for all curves except grumpkin:

```go
// ECNTT performs Elliptic Curve Number Theoretic Transform
type ECNTT struct {
    // ... internal fields
}

// Forward performs forward ECNTT
func (e *ECNTT) Forward(points []Point) ([]Point, error)

// Inverse performs inverse ECNTT
func (e *ECNTT) Inverse(points []Point) ([]Point, error)

// BatchForward performs batch forward ECNTT
func (e *ECNTT) BatchForward(points []Point, batchSize int) ([]Point, error)

// BatchInverse performs batch inverse ECNTT
func (e *ECNTT) BatchInverse(points []Point, batchSize int) ([]Point, error)
```

## Usage Example

Here's an example of how to use curve operations:

```go
package main

import (
    "fmt"
    "icicle/curves/bn254"
)

func main() {
    // Create points
    p1 := &bn254.Point{
        X: bn254.NewField(1),
        Y: bn254.NewField(2),
    }
    p2 := &bn254.Point{
        X: bn254.NewField(3),
        Y: bn254.NewField(4),
    }

    // Add points
    sum, err := p1.Add(p2)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Sum: (%v, %v)\n", sum.X, sum.Y)
}
```

## Supported Operations by Curve

| Operation\Curve | bn254 | bls12_377 | bls12_381 | bw6-761 | grumpkin |
| --- | :---: | :---: | :---: | :---: | :---: |
| Point Addition | ✅ | ✅ | ✅ | ✅ | ✅ |
| Scalar Multiplication | ✅ | ✅ | ✅ | ✅ | ✅ |
| G2 Operations | ✅ | ✅ | ✅ | ✅ | ❌ |
| ECNTT | ✅ | ✅ | ✅ | ✅ | ❌ |
| Pairing | ✅ | ✅ | ✅ | ✅ | ❌ |

## Error Handling

All curve operations return an error that should be checked:

```go
sum, err := p1.Add(p2)
if err != nil {
    // Handle error
}
```

Common errors include:
- Invalid point coordinates
- Points not on curve
- Invalid scalar values
- Memory allocation failures
- Device errors

## Performance Considerations

1. Use batch operations when possible to amortize overhead
2. Keep points in projective coordinates for better performance
3. Use Montgomery form for field arithmetic
4. Consider using G2 operations only when necessary
5. Use ECNTT for large batches of point operations 