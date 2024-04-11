package babybear
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
	LIMBS int8 = 1
)

type Field struct {
	limbs [LIMBS]uint32
}

func (f Field) Len() int {
	return int(LIMBS)
}

func (f Field) Size() int {
	return int(LIMBS * 4)
}

func (f Field) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f Field) AsPointer() *uint32 {
	return &f.limbs[0]
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

func GenerateScalars(size int) core.HostSlice[Field] {
	scalarSlice := make(core.HostSlice[Field], size)

	cScalars := (*C.scalar_t)(unsafe.Pointer(&scalarSlice[0]))
	cSize := (C.int)(size)
	C.babybearGenerateScalars(cScalars, cSize)

	return scalarSlice
}

func convertScalarsMontgomery(scalars *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C.scalar_t)(scalars.AsPointer())
	cSize := (C.size_t)(scalars.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.babybearScalarConvertMontgomery(cValues, cSize, cIsInto, cCtx)
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
