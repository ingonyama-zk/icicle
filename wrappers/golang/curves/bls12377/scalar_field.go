package bls12377

// #cgo CFLAGS: -I./include/
// #include "scalar_field.h"
import "C"
import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

const (
	SCALAR_LIMBS int = 4
)

type ScalarField struct {
	limbs [SCALAR_LIMBS]uint64
}

func (f ScalarField) Len() int {
	return int(SCALAR_LIMBS)
}

func (f ScalarField) Size() int {
	return int(SCALAR_LIMBS * 8)
}

func (f ScalarField) GetLimbs() []uint64 {
	return f.limbs[:]
}

func (f ScalarField) AsPointer() *uint64 {
	return &f.limbs[0]
}

func (f *ScalarField) FromLimbs(limbs []uint64) ScalarField {
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
	if len(bytes)/8 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*8, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint64(bytes[i*8 : i*8+8])
	}

	return *f
}

func (f ScalarField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*8)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint64(bytes[i*8:], v)
	}

	return bytes
}

func GenerateScalars(size int) core.HostSlice[ScalarField] {
	scalarSlice := make(core.HostSlice[ScalarField], size)

	cScalars := (*C.scalar_t)(unsafe.Pointer(&scalarSlice[0]))
	cSize := (C.int)(size)
	C.bls12_377GenerateScalars(cScalars, cSize)

	return scalarSlice
}

func convertScalarsMontgomery(scalars *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C.scalar_t)(scalars.AsUnsafePointer())
	cSize := (C.size_t)(scalars.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bls12_377ScalarConvertMontgomery(cValues, cSize, cIsInto, cCtx)
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
