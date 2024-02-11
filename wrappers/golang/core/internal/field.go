package internal

import (
	"encoding/binary"
	"fmt"
)

const (
	BASE_LIMBS int8 = 8
)

type Field struct {
	limbs [BASE_LIMBS]uint32
}

func (f Field) Len() int {
	return int(BASE_LIMBS)
}

func (f Field) Size() int {
	return int(BASE_LIMBS*4)
}

func (f Field) AsPointer() *uint32 {
	return &f.limbs[0]
}

func (f Field) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f *Field) FromLimbs(limbs []uint32) Field {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *Field) Zero() Field {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	
	return *f
}

func (f *Field) One() Field {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *Field) FromBytesLittleEndian(bytes []byte) Field {
	if len(bytes)/4 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
	}

	return *f
}

func (f Field) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*4)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}
