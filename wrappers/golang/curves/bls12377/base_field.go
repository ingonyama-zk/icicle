package bls12377

import (
	"encoding/binary"
	"fmt"
)

const (
	BASE_LIMBS int = 12
)

type BaseField struct {
	limbs [BASE_LIMBS]uint32
}

func (f BaseField) Len() int {
	return int(BASE_LIMBS)
}

func (f BaseField) Size() int {
	return int(BASE_LIMBS * 4)
}

func (f BaseField) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f BaseField) AsPointer() *uint32 {
	return &f.limbs[0]
}

func (f *BaseField) FromUint32(v uint32) BaseField {
	f.limbs[BASE_LIMBS - 1] = v
	return *f
}

func (f *BaseField) FromLimbs(limbs []uint32) BaseField {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *BaseField) Zero() BaseField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}

	return *f
}

func (f *BaseField) One() BaseField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *BaseField) FromBytesLittleEndian(bytes []byte) BaseField {
	if len(bytes)/4 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
	}

	return *f
}

func (f BaseField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*4)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}

