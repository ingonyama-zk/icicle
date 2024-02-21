package bls12381

// #cgo CFLAGS: -I./include/
// #include "ntt.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func GetDefaultNttConfig() core.NTTConfig[[SCALAR_LIMBS]uint32] {
	cosetGenField := ScalarField{}
	cosetGenField.One()
	var cosetGen [SCALAR_LIMBS]uint32
	for i, v := range cosetGenField.GetLimbs() {
		cosetGen[i] = v
	}

	return core.GetDefaultNTTConfig(cosetGen)
}

func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) core.IcicleError {
	core.NttCheck[T](scalars, cfg, results)

	var scalarsPointer unsafe.Pointer
	if scalars.IsOnDevice() {
		scalarsPointer = scalars.(core.DeviceSlice).AsPointer()
	} else {
		scalarsPointer = unsafe.Pointer(&scalars.(core.HostSlice[ScalarField])[0])
	}
	cScalars := (*C.scalar_t)(scalarsPointer)
	cSize := (C.int)(scalars.Len() / int(cfg.BatchSize))
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(unsafe.Pointer(cfg))

	var resultsPointer unsafe.Pointer
	if results.IsOnDevice() {
		resultsPointer = results.(core.DeviceSlice).AsPointer()
	} else {
		resultsPointer = unsafe.Pointer(&results.(core.HostSlice[ScalarField])[0])
	}
	cResults := (*C.scalar_t)(resultsPointer)

	__ret := C.bls12_381NTTCuda(cScalars, cSize, cDir, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func InitDomain(primitiveRoot ScalarField, ctx cr.DeviceContext) core.IcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(primitiveRoot.AsPointer()))
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	__ret := C.bls12_381InitializeDomain(cPrimitiveRoot, cCtx)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
