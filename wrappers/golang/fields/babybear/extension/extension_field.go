package extension

// #cgo CFLAGS: -I./include/
// #include "scalar_field.h"
import "C"
import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

const (
	EXTENSION_LIMBS int = 4
)

type ExtensionField struct {
	limbs [EXTENSION_LIMBS]uint32
}

func (f ExtensionField) Len() int {
	return int(EXTENSION_LIMBS)
}

func (f ExtensionField) Size() int {
	return int(EXTENSION_LIMBS * 4)
}

func (f ExtensionField) GetLimbs() []uint32 {
	return f.limbs[:]
}

func (f ExtensionField) AsPointer() *uint32 {
	return &f.limbs[0]
}

func (f *ExtensionField) FromUint32(v uint32) ExtensionField {
	f.limbs[0] = v
	return *f
}

func (f *ExtensionField) FromLimbs(limbs []uint32) ExtensionField {
	if len(limbs) != f.Len() {
		panic("Called FromLimbs with limbs of different length than field")
	}
	for i := range f.limbs {
		f.limbs[i] = limbs[i]
	}

	return *f
}

func (f *ExtensionField) Zero() ExtensionField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}

	return *f
}

func (f *ExtensionField) One() ExtensionField {
	for i := range f.limbs {
		f.limbs[i] = 0
	}
	f.limbs[0] = 1

	return *f
}

func (f *ExtensionField) FromBytesLittleEndian(bytes []byte) ExtensionField {
	if len(bytes)/4 != f.Len() {
		panic(fmt.Sprintf("Called FromBytesLittleEndian with incorrect bytes length; expected %d - got %d", f.Len()*4, len(bytes)))
	}

	for i := range f.limbs {
		f.limbs[i] = binary.LittleEndian.Uint32(bytes[i*4 : i*4+4])
	}

	return *f
}

func (f ExtensionField) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.Len()*4)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}

	return bytes
}

func GenerateScalars(size int) core.HostSlice[ExtensionField] {
	scalarSlice := make(core.HostSlice[ExtensionField], size)

	cScalars := (*C.scalar_t)(unsafe.Pointer(&scalarSlice[0]))
	cSize := (C.int)(size)
	C.babybear_extension_generate_scalars(cScalars, cSize)

	return scalarSlice
}

func convertScalarsMontgomery(scalars core.HostOrDeviceSlice, isInto bool) runtime.EIcicleError {
	defaultCfg := core.DefaultVecOpsConfig()
	cValues, _, _, cCfg, cSize := core.VecOpCheck(scalars, scalars, scalars, &defaultCfg)
	cErr := C.babybear_extension_scalar_convert_montgomery((*C.scalar_t)(cValues), (C.size_t)(cSize), (C._Bool)(isInto), (*C.VecOpsConfig)(cCfg), (*C.scalar_t)(cValues))
	err := runtime.EIcicleError(cErr)
	return err
}

func ToMontgomery(scalars core.HostOrDeviceSlice) runtime.EIcicleError {
	return convertScalarsMontgomery(scalars, true)
}

func FromMontgomery(scalars core.HostOrDeviceSlice) runtime.EIcicleError {
	return convertScalarsMontgomery(scalars, false)
}
