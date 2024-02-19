package bn254

// // #cgo CFLAGS: -I./include/
// // #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254
// // #include "field.h"
// import "C"
// import (
// 	// core "github.com/ingonyama-zk/icicle/wrappers/golang/core"
// 	// cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
// 	// "unsafe"
// 	"encoding/binary"
// )

// const (
// 	G2_BASE_LIMBS 		int8 = 16
// )

// type G2BaseField struct {
// 	limbs [G2_BASE_LIMBS]uint32
// }

// func (f G2BaseField) Size() int {
// 	return int(G2_BASE_LIMBS*4)
// }

// func (f G2BaseField) AsPointer() *uint32 {
// 	return &f.limbs[0]
// }

// func (f *G2BaseField) GetLimbs() []uint32 {
// 	return f.limbs[:]
// }

// func (f* G2BaseField) FromLimbs(limbs []uint32) G2BaseField {
// 	for i := range f.limbs {
// 		f.limbs[i] = limbs[i]
// 	}

// 	return *f
// }

// func (f* G2BaseField) Zero() {
// 	for i := range f.limbs {
// 		f.limbs[i] = 0
// 	}
// }

// func (f* G2BaseField) One() {
// 	for i := range f.limbs {
// 		f.limbs[i] = 0
// 	}
// 	f.limbs[0] = 1
// }

// func (f* G2BaseField) FromBytesLittleEndian(bytes []byte) G2BaseField {
// 	for i := range f.limbs {
// 		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4:i*4+4])
// 	}

// 	return *f
// }

// func (f* G2BaseField) ToBytesLittleEndian() []byte {
// 	bytes := make([]byte, len(f.limbs)*4)
// 	for i, v := range f.limbs {
// 		binary.LittleEndian.PutUint32(bytes[i*4:], v)
// 	}

// 	return bytes
// }
