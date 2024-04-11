package internal

import (
	"encoding/binary"
	"fmt"
)

const (
	BASE_LIMBS int = 4
)

type MockField struct {
	limbs [BASE_LIMBS]uint64
}

func (f MockField) Len() int {
	return int(BASE_LIMBS)
}

func (f MockField) Size() int {
	return int(BASE_LIMBS * 8)
}

func (f MockField) GetLimbs() []uint64 {
	return f.limbs[:]
}

func (f MockField) AsPointer() *uint64 {
	return &f.limbs[0]
}

func (f *MockField) FromLimbs(limbs []uint64) MockField {
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
	if len(bytes)/8 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*8, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint64(bytes[i*8 : i*8+8])
	}

	return *f
}

func (f MockField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*8)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint64(bytes[i*8:], v)
	}

	return bytes
}
