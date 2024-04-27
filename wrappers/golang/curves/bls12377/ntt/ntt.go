package ntt

// #cgo CFLAGS: -I./include/
// #include "ntt.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"
)

func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) core.IcicleError {
	scalarsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](scalars, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.scalar_t)(resultsPointer)

	__ret := C.bls12_377_ntt_cuda(cScalars, cSize, cDir, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func GetDefaultNttConfig() core.NTTConfig[[bls12_377.SCALAR_LIMBS]uint32] {
	cosetGenField := bls12_377.ScalarField{}
	cosetGenField.One()
	var cosetGen [bls12_377.SCALAR_LIMBS]uint32
	for i, v := range cosetGenField.GetLimbs() {
		cosetGen[i] = v
	}

	return core.GetDefaultNTTConfig(cosetGen)
}

func InitDomain(primitiveRoot bls12_377.ScalarField, ctx cr.DeviceContext, fastTwiddles bool) core.IcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(primitiveRoot.AsPointer()))
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	cFastTwiddles := (C._Bool)(fastTwiddles)
	__ret := C.bls12_377_initialize_domain(cPrimitiveRoot, cCtx, cFastTwiddles)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func ReleaseDomain(ctx cr.DeviceContext) core.IcicleError {
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	__ret := C.bls12_377_release_domain(cCtx)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
