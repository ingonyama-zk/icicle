package bn254

import (
	"unsafe"

	"encoding/binary"
	"fmt"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// #cgo CFLAGS: -I${SRCDIR}/icicle/curves/bn254/
// #cgo LDFLAGS: -L${SRCDIR}/icicle/curves/bn254/ -lbn254
// #include "c_api.h"
import "C"

const SIZE = 8

type ScalarField struct {
	s [SIZE]uint32
}

type BaseField struct {
	s [SIZE]uint32
}

type Field interface {
	ScalarField | BaseField
}

/*
 * Common Constrctors
 */

func NewFieldZero[T Field]() *T {
	var field T

	return &field
}

func NewFieldOne[T Field]() *T {
	var s [SIZE]uint32

	s[0] = 1

	return &T{s}
}

func NewFieldFromFrGnark[T Field](element fr.Element) *T {
	s := ConvertUint64ArrToUint32Arr(element.Bits()) // get non-montgomry

	return &T{s}
}

func NewFieldFromFpGnark[T BaseField | ScalarField](element fp.Element) *T {
	s := ConvertUint64ArrToUint32Arr(element.Bits()) // get non-montgomry

	return &T{s}
}

/*
 * BaseField Constrctors
 */

func BaseFieldFromLimbs(limbs [8]uint32) *BaseField {
	bf := NewFieldZero[BaseField]()
	copy(bf.s[:], limbs[:])

	return bf
}

/*
 * BaseField methods
 */

func (f *BaseField) limbs() [SIZE]uint32 {
	return f.s
}

func (f *BaseField) toBytesLe() []byte {
	bytes := make([]byte, len(f.s)*4)
	for i, v := range f.s {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}

func (f *BaseField) toGnarkFr() *fr.Element {
	fb := f.toBytesLe()
	var b32 [32]byte
	copy(b32[:], fb[:32])

	v, e := fr.LittleEndian.Element(&b32)

	if e != nil {
		panic(fmt.Sprintf("unable to create convert point %v got error %v", f, e))
	}

	return &v
}

func (f *BaseField) toGnarkFp() *fp.Element {
	fb := f.toBytesLe()
	var b32 [32]byte
	copy(b32[:], fb[:32])

	v, e := fp.LittleEndian.Element(&b32)

	if e != nil {
		panic(fmt.Sprintf("unable to create convert point %v got error %v", f, e))
	}

	return &v
}

/*
 * ScalarField methods
 */

func (f *ScalarField) limbs() [SIZE]uint32 {
	return f.s
}

func (f *ScalarField) toBytesLe() []byte {
	bytes := make([]byte, len(f.s)*4)
	for i, v := range f.s {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}

func (f *ScalarField) toGnarkFr() *fr.Element {
	fb := f.toBytesLe()
	var b32 [32]byte
	copy(b32[:], fb[:32])

	v, e := fr.LittleEndian.Element(&b32)

	if e != nil {
		panic(fmt.Sprintf("unable to create convert point %v got error %v", f, e))
	}

	return &v
}

func (f *ScalarField) toGnarkFp() *fp.Element {
	fb := f.toBytesLe()
	var b32 [32]byte
	copy(b32[:], fb[:32])

	v, e := fp.LittleEndian.Element(&b32)

	if e != nil {
		panic(fmt.Sprintf("unable to create convert point %v got error %v", f, e))
	}

	return &v
}

/*
 * PointBN254
 */

type PointBN254 struct {
	x, y, z BaseField
}

func NewPointBN254Zero() *PointBN254 {
	return &PointBN254{
		x: *NewFieldZero[BaseField](),
		y: *NewFieldOne[BaseField](),
		z: *NewFieldZero[BaseField](),
	}
}

func (p *PointBN254) eq(pCompare *PointBN254) bool {
	// Cast *PointBN254 to *C.BN254_projective_t
	// The unsafe.Pointer cast is necessary because Go doesn't allow direct casts
	// between different pointer types.
	// It's your responsibility to ensure that the types are compatible.
	pC := (*C.BN254_projective_t)(unsafe.Pointer(p))
	pCompareC := (*C.BN254_projective_t)(unsafe.Pointer(pCompare))

	// Call the C function
	// The C function doesn't keep any references to the data,
	// so it's fine if the Go garbage collector moves or deletes the data later.
	return bool(C.eq_bn254(pC, pCompareC, 0))
}

func (p *PointBN254) strip_z() *PointAffineNoInfinityBN254 {
	return &PointAffineNoInfinityBN254{
		x: p.x,
		y: p.y,
	}
}

func (p *PointBN254) toGnarkAffine() *bn254.G1Affine {
	px := p.x.toGnarkFp()
	py := p.y.toGnarkFp()
	pz := p.z.toGnarkFp()

	zInv := new(fp.Element)
	x := new(fp.Element)
	y := new(fp.Element)

	zInv.Inverse(pz)

	x.Mul(px, zInv)
	y.Mul(py, zInv)

	return &bn254.G1Affine{X: *x, Y: *y}
}

// converts jac fromat to projective
func PointBN254FromGnark(gnark *bn254.G1Jac) *PointBN254 {
	var pointAffine bn254.G1Affine
	pointAffine.FromJacobian(gnark)

	point := PointBN254{
		x: *NewFieldFromFpGnark[BaseField](pointAffine.X),
		y: *NewFieldFromFpGnark[BaseField](pointAffine.Y),
		z: *NewFieldOne[BaseField](),
	}

	return &point
}

func PointBN254fromLimbs(x, y, z *[]uint32) *PointBN254 {
	return &PointBN254{
		x: *BaseFieldFromLimbs(getFixedLimbs(x)),
		y: *BaseFieldFromLimbs(getFixedLimbs(y)),
		z: *BaseFieldFromLimbs(getFixedLimbs(z)),
	}
}

/*
 * PointAffineNoInfinityBN254
 */

type PointAffineNoInfinityBN254 struct {
	x, y BaseField
}

func NewPointAffineNoInfinityBN254Zero() *PointAffineNoInfinityBN254 {
	return &PointAffineNoInfinityBN254{
		x: *NewFieldZero[BaseField](),
		y: *NewFieldZero[BaseField](),
	}
}

func (p *PointAffineNoInfinityBN254) toProjective() *PointBN254 {
	return &PointBN254{
		x: p.x,
		y: p.y,
		z: *NewFieldOne[BaseField](),
	}
}

func (p *PointAffineNoInfinityBN254) toGnarkAffine() *bn254.G1Affine {
	return p.toProjective().toGnarkAffine()
}

func PointAffineNoInfinityBN254FromLimbs(x, y *[]uint32) *PointAffineNoInfinityBN254 {
	return &PointAffineNoInfinityBN254{
		x: *BaseFieldFromLimbs(getFixedLimbs(x)),
		y: *BaseFieldFromLimbs(getFixedLimbs(y)),
	}
}

/*
 * Utils
 */

func getFixedLimbs(slice *[]uint32) [8]uint32 {
	if len(*slice) <= 8 {
		limbs := [8]uint32{}
		copy(limbs[:len(*slice)], *slice)
		return limbs
	}

	panic("slice has too many elements")
}
