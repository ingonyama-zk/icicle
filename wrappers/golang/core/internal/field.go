package internal

import (
	"encoding/binary"
	"fmt"
)

const (
	BASE_LIMBS int8 = 8
)

type MockField struct {
	limbs [BASE_LIMBS]uint32
}

func (f MockField) Len() int {
	return int(BASE_LIMBS)
}

func (f MockField) Size() int {
	return int(BASE_LIMBS * 4)
}

func (f MockField) AsPointer() *uint32 {
	return &f.limbs[0]
}

func (f MockField) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f *MockField) FromLimbs(limbs []uint32) MockField {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *MockField) Zero() MockField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}

	return *f
}

func (f *MockField) One() MockField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *MockField) FromBytesLittleEndian(bytes []byte) MockField {
	if len(bytes)/4 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
	}

	return *f
}

func (f MockField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*4)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}
