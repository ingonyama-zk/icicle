package ntt

// #cgo CFLAGS: -I./include/
// #include "ntt.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	babybear "github.com/ingonyama-zk/icicle/v2/wrappers/golang/fields/babybear"
)

import (
	"unsafe"
)

func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) core.IcicleError {
	scalarsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](scalars, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.scalar_t)(resultsPointer)

	__ret := C.babybear_ntt_cuda(cScalars, cSize, cDir, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func GetDefaultNttConfig() core.NTTConfig[[babybear.SCALAR_LIMBS]uint32] {
	cosetGenField := babybear.ScalarField{}
	cosetGenField.One()
	var cosetGen [babybear.SCALAR_LIMBS]uint32
	for i, v := range cosetGenField.GetLimbs() {
		cosetGen[i] = v
	}

	return core.GetDefaultNTTConfig(cosetGen)
}

func InitDomain(primitiveRoot babybear.ScalarField, ctx cr.DeviceContext, fastTwiddles bool) core.IcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(primitiveRoot.AsPointer()))
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	cFastTwiddles := (C._Bool)(fastTwiddles)
	__ret := C.babybear_initialize_domain(cPrimitiveRoot, cCtx, cFastTwiddles)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func ReleaseDomain(ctx cr.DeviceContext) core.IcicleError {
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	__ret := C.babybear_release_domain(cCtx)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
