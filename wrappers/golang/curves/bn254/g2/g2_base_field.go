package g2

import (
	"encoding/binary"
	"fmt"
)

const (
	G2_BASE_LIMBS int = 8
)

type G2BaseField struct {
	limbs [G2_BASE_LIMBS]uint64
}

func (f G2BaseField) Len() int {
	return int(G2_BASE_LIMBS)
}

func (f G2BaseField) Size() int {
	return int(G2_BASE_LIMBS * 8)
}

func (f G2BaseField) GetLimbs() []uint64 {
	return f.limbs[:]
}

func (f G2BaseField) AsPointer() *uint64 {
	return &f.limbs[0]
}

func (f *G2BaseField) FromLimbs(limbs []uint64) G2BaseField {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *G2BaseField) Zero() G2BaseField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}

	return *f
}

func (f *G2BaseField) One() G2BaseField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *G2BaseField) FromBytesLittleEndian(bytes []byte) G2BaseField {
	if len(bytes)/8 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*8, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint64(bytes[i*8 : i*8+8])
	}

	return *f
}

func (f G2BaseField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*8)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint64(bytes[i*8:], v)
	}

	return bytes
}
