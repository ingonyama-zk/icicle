package core

import (
	"encoding/binary"
)

type FieldInter interface {
	GetLimbs() []uint32
	FromLimbs(limbs []uint32) Field
	Zero()
	One()
	FromBytesLittleEndian(bytes []byte) Field
	ToBytesLittleEndian() []byte
}

type Field struct {
	Limbs []uint32
}

func (f* Field) Size() int {
	return len(f.Limbs)*4
}

func (f* Field) GetLimbs() []uint32 {
	return (f.Limbs)
}

func (f* Field) FromLimbs(limbs []uint32) Field {
	f.Limbs = limbs

	return *f
}

func (f* Field) Zero() {
	for i, _ := range f.Limbs {
		f.Limbs[i] = 0
	}
}

func (f* Field) One() {
	for i, _ := range f.Limbs {
		f.Limbs[i] = 0
	}
	f.Limbs[0] = 1
}

func (f* Field) FromBytesLittleEndian(bytes []byte) Field {
	for i, _ := range f.Limbs {
		f.Limbs[i] = binary.LittleEndian.Uint32(bytes[i:i+4])
	}

	return *f
}

func (f* Field) ToBytesLittleEndian() []byte {
	bytes := make([]byte, len(f.Limbs)*4)
	for i, v := range f.Limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}
