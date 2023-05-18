package bn254

import (
	"encoding/binary"
)

// #cgo CFLAGS: -I../../icicle/curves/bn254/
// #cgo LDFLAGS: -L../../icicle/curves/bn254/ -lbn254
// #include "c_api.h"
import "C"

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

type BaseFieldBN254 interface {
	limbs() [8]uint32
	toBytesLe() []byte
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

type BasePointBN254 interface {
	eq(*PointBN254) bool
}

func (p *PointBN254) eq(pCompare *PointBN254) bool {
	return bool(C.eq_bn254(p, pCompare, 0))
}

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

/**
func (p *PointAffineNoInfinityBN254) fromLimbs(x, y *[]uint32) *PointAffineNoInfinityBN254 {
	return &PointAffineNoInfinityBN254{
		x: FieldBN254{ s }
		y:
	}
}

func getFixedLimbs
*/
