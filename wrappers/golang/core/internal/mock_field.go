package internal

import (
	"encoding/binary"
	"fmt"
)

const (
	MOCKBASE_LIMBS     int = 4
	MockBaseFieldBytes int = MOCKBASE_LIMBS * 4
)

type MockBaseField struct {
	limbs [MOCKBASE_LIMBS]uint32
}

func (f MockBaseField) Len() int {
	return int(MOCKBASE_LIMBS)
}

func (f MockBaseField) Size() int {
	return int(MOCKBASE_LIMBS * 4)
}

func (f MockBaseField) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f MockBaseField) AsPointer() *uint32 {
	return &f.limbs[0]
}

func (f *MockBaseField) FromUint32(v uint32) MockBaseField {
	f.limbs[0] = v
	return *f
}

func (f *MockBaseField) FromLimbs(limbs []uint32) MockBaseField {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *MockBaseField) Zero() MockBaseField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}

	return *f
}

func (f *MockBaseField) One() MockBaseField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *MockBaseField) FromBytesLittleEndian(bytes []byte) MockBaseField {
	if len(bytes)/4 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
	}

	return *f
}

func (f MockBaseField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*4)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}
