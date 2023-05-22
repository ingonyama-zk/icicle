package bn254

import (
	"encoding/binary"
	"unsafe"
)

// #cgo CFLAGS: -I../../../icicle/curves/bn254/
// #cgo LDFLAGS: -L../../../icicle/curves/bn254/ -lbn254
// #include "c_api.h"
import "C"

/*
 * FieldBN254
 */

type FieldBN254 struct {
	// todo make 8 gloabl const
	s [8]uint32
}

func NewFieldBN254One() *FieldBN254 {
	var s [8]uint32
	s[0] = 1

	return &FieldBN254{s}
}

func NewFieldBN254Zero() *FieldBN254 {
	var field FieldBN254

	return &field
}

func (f *FieldBN254) limbs() [8]uint32 {
	return f.s
}

func (f *FieldBN254) toBytesLe() []byte {
	bytes := make([]byte, len(f.s)*4) // each uint32 takes 4 bytes
	for i, v := range f.s {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}

/*
 * PointBN254
 */

type PointBN254 struct {
	x, y, z FieldBN254
}

func NewPointBN254Zero() *PointBN254 {
	return &PointBN254{
		x: *NewFieldBN254Zero(),
		y: *NewFieldBN254One(),
		z: *NewFieldBN254Zero(),
	}
}

func NewPointBN254Infinity() *PointBN254 {
	return &PointBN254{
		x: *NewFieldBN254Zero(),
		y: *NewFieldBN254One(),
		z: *NewFieldBN254Zero(),
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

func PointBN254fromLimbs(x, y, z *[]uint32) *PointBN254 {
	return &PointBN254{
		x: FieldBN254{s: getFixedLimbs(x)},
		y: FieldBN254{s: getFixedLimbs(y)},
		z: FieldBN254{s: getFixedLimbs(z)},
	}
}

/*
 * PointAffineNoInfinityBN254
 */

type PointAffineNoInfinityBN254 struct {
	x, y FieldBN254
}

func NewPointAffineNoInfinityBN254Zero() *PointAffineNoInfinityBN254 {
	return &PointAffineNoInfinityBN254{
		x: *NewFieldBN254Zero(),
		y: *NewFieldBN254Zero(),
	}
}

func (p *PointAffineNoInfinityBN254) limbs() []uint32 {
	sliceX := p.x.limbs()
	sliceY := p.y.limbs()

	return append(sliceX[:], sliceY[:]...)
}

func (p *PointAffineNoInfinityBN254) toProjective() *PointBN254 {
	return &PointBN254{
		x: p.x,
		y: p.y,
		z: *NewFieldBN254One(),
	}
}

func PointAffineNoInfinityBN254FromLimbs(x, y *[]uint32) *PointAffineNoInfinityBN254 {
	return &PointAffineNoInfinityBN254{
		x: FieldBN254{s: getFixedLimbs(x)},
		y: FieldBN254{s: getFixedLimbs(y)},
	}
}

/*
 * Utils
 */

func getFixedLimbs(slice *[]uint32) [8]uint32 {
	if len(*slice) < 8 {
		limbs := [8]uint32(*slice)
		return limbs
	}

	panic("slice has too many elements")
}
