package g2

// import (
// 	"encoding/binary"
// 	"fmt"
// )

// const (
// 	G2BASE_LIMBS int = 16
// )

// type G2BaseField struct {
// 	limbs [G2BASE_LIMBS]uint32
// }

// func (f G2BaseField) Len() int {
// 	return int(G2BASE_LIMBS)
// }

// func (f G2BaseField) Size() int {
// 	return int(G2BASE_LIMBS * 4)
// }

// func (f G2BaseField) GetLimbs() []uint32 {
// 	return f.limbs[:]
// }

// func (f G2BaseField) AsPointer() *uint32 {
// 	return &f.limbs[0]
// }

// func (f *G2BaseField) FromUint32(v uint32) G2BaseField {
// 	f.limbs[0] = v
// 	return *f
// }

// func (f *G2BaseField) FromLimbs(limbs []uint32) G2BaseField {
// 	if len(limbs) != f.Len() {
// 		panic("Called FromLimbs with limbs of different length than field")
// 	}
// 	for i := range f.limbs {
// 		f.limbs[i] = limbs[i]
// 	}

// 	return *f
// }

// func (f *G2BaseField) Zero() G2BaseField {
// 	for i := range f.limbs {
// 		f.limbs[i] = 0
// 	}

// 	return *f
// }

// func (f *G2BaseField) One() G2BaseField {
// 	for i := range f.limbs {
// 		f.limbs[i] = 0
// 	}
// 	f.limbs[0] = 1

// 	return *f
// }

// func (f *G2BaseField) FromBytesLittleEndian(bytes []byte) G2BaseField {
// 	if len(bytes)/4 != f.Len() {
// 		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
// 	}

// 	for i := range f.limbs {
// 		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
// 	}

// 	return *f
// }

// func (f G2BaseField) ToBytesLittleEndian() []byte {
// 	bytes := make([]byte, f.Len()*4)
// 	for i, v := range f.limbs {
// 		binary.LittleEndian.PutUint32(bytes[i*4:], v)
// 	}

// 	return bytes
// }
