package bls12381

// #cgo CFLAGS: -I./include/
// #include "scalar_field.h"
import "C"
import (
	"encoding/binary"
	"fmt"
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	"unsafe"
)

const (
	SCALAR_LIMBS int = 8
)

type ScalarField struct {
	limbs [SCALAR_LIMBS]uint32
}

func (f ScalarField) Len() int {
	return int(SCALAR_LIMBS)
}

func (f ScalarField) Size() int {
	return int(SCALAR_LIMBS * 4)
}

func (f ScalarField) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f ScalarField) AsPointer() *uint32 {
	return &f.limbs[0]
}

func (f *ScalarField) FromUint32(v uint32) ScalarField {
	f.limbs[0] = v
	return *f
}

func (f *ScalarField) FromLimbs(limbs []uint32) ScalarField {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *ScalarField) Zero() ScalarField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}

	return *f
}

func (f *ScalarField) One() ScalarField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *ScalarField) FromBytesLittleEndian(bytes []byte) ScalarField {
	if len(bytes)/4 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
	}

	return *f
}

func (f ScalarField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*4)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}

func GenerateScalars(size int) core.HostSlice[ScalarField] {
	scalarSlice := make(core.HostSlice[ScalarField], size)

	cScalars := (*C.scalar_t)(unsafe.Pointer(&scalarSlice[0]))
	cSize := (C.int)(size)
	C.bls12_381_generate_scalars(cScalars, cSize)

	return scalarSlice
}

func convertScalarsMontgomery(scalars *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C.scalar_t)(scalars.AsUnsafePointer())
	cSize := (C.size_t)(scalars.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bls12_381_scalar_convert_montgomery(cValues, cSize, cIsInto, cCtx)
	err := (cr.CudaError)(__ret)
	return err
}

func ToMontgomery(scalars *core.DeviceSlice) cr.CudaError {
	scalars.CheckDevice()
	return convertScalarsMontgomery(scalars, true)
}

func FromMontgomery(scalars *core.DeviceSlice) cr.CudaError {
	scalars.CheckDevice()
	return convertScalarsMontgomery(scalars, false)
}
