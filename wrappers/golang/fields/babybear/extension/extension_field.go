package extension

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
	EXTENSION_LIMBS int = 4
)

type extensionField struct {
	limbs [EXTENSION_LIMBS]uint32
}

func (f extensionField) Len() int {
	return int(EXTENSION_LIMBS)
}

func (f extensionField) Size() int {
	return int(EXTENSION_LIMBS * 4)
}

func (f extensionField) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f extensionField) AsPointer() *uint32 {
	return &f.limbs[0]
}

func (f *extensionField) FromUint32(v uint32) extensionField {
	f.limbs[EXTENSION_LIMBS-1] = v
	return *f
}

func (f *extensionField) FromLimbs(limbs []uint32) extensionField {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *extensionField) Zero() extensionField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}

	return *f
}

func (f *extensionField) One() extensionField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *extensionField) FromBytesLittleEndian(bytes []byte) extensionField {
	if len(bytes)/4 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
	}

	return *f
}

func (f extensionField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*4)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}

func GenerateScalars(size int) core.HostSlice[extensionField] {
	scalarSlice := make(core.HostSlice[extensionField], size)

	cScalars := (*C.scalar_t)(unsafe.Pointer(&scalarSlice[0]))
	cSize := (C.int)(size)
	C.babybear_extension_generate_scalars(cScalars, cSize)

	return scalarSlice
}

func convertScalarsMontgomery(scalars *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C.scalar_t)(scalars.AsUnsafePointer())
	cSize := (C.size_t)(scalars.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.babybear_extension_scalar_convert_montgomery(cValues, cSize, cIsInto, cCtx)
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
