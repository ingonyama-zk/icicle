package bn254

// #cgo CFLAGS: -I./include/
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254
// #include "field.h"
import "C"
import (
	core "local/hello/icicle/wrappers/golang/core"
	cr "local/hello/icicle/wrappers/golang/cuda_runtime"
	"unsafe"
)

const (
	SCALAR_LIMBS 		int8 = 8
	BASE_LIMBS   		int8 = 8
	G2_BASE_LIMBS   int8 = 16
)

// TODO: figure out why the size is 24 and not 32
// Its because we were measuring the size of the slice and not the underlying data
func newBN254ScalarField() core.Field {
	sField :=	&core.Field{
		Limbs: make([]uint32, SCALAR_LIMBS),
	}
	sField.Zero()
	return *sField
}

func newBN254Field() core.Field {
	bField :=	&core.Field{
		Limbs: make([]uint32, BASE_LIMBS),
	}
	bField.Zero()
	return *bField
}

func newG2BN254Field() core.Field {
	bG2Field :=	&core.Field{
		Limbs: make([]uint32, G2_BASE_LIMBS),
	}
	bG2Field.Zero()
	return *bG2Field
}

func GenerateScalars(size int) core.HostSlice {
	scalars := make(core.HostSlice, size)
	for i := range scalars {
		scalars[i] = newBN254ScalarField()
	}

	cScalars := (*C.scalar_t)(unsafe.Pointer(&(scalars[0].Limbs[0])))
	cSize := (C.int)(size)
	C.bn254GenerateScalars(cScalars, cSize)

	return scalars
}

func convertScalarsMontgomery[T any, S any](scalars cr.HostOrDeviceSlice[T, S], isInto bool) cr.CudaError {
	scalarPointer := scalars.AsPointer()
	cValues := (*C.scalar_t)(unsafe.Pointer(scalarPointer))
	cSize := (C.size_t)(scalars.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bn254ScalarConvertMontgomery(cValues, cSize, cIsInto, cCtx) // template this per curve
	err := (cr.CudaError)(__ret)
	return err
}

func convertScalarsMontgomeryUnsafe(scalars unsafe.Pointer, size int, isInto bool) cr.CudaError {
	cValues := (*C.scalar_t)(scalars)
	cSize := (C.size_t)(size)
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bn254ScalarConvertMontgomery(cValues, cSize, cIsInto, cCtx) // template this per curve
	err := (cr.CudaError)(__ret)
	return err
}

func ToMontgomery[T any, S any](scalars cr.HostOrDeviceSlice[T, S]) cr.CudaError {
	return convertScalarsMontgomery(scalars, true)
}

func ToMontgomeryUnsafe(scalars unsafe.Pointer, size int) cr.CudaError {
	return convertScalarsMontgomeryUnsafe(scalars, size, true)
}

func FromMontgomery[T any, S any](scalars cr.HostOrDeviceSlice[T, S]) cr.CudaError {
	return convertScalarsMontgomery(scalars, false)
}
