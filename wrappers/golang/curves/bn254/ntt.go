package bn254

// #cgo CFLAGS: -I./include/
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254
// #include "ntt.h"
import "C"

import (
	"local/hello/icicle/wrappers/golang/core"
	cr "local/hello/icicle/wrappers/golang/cuda_runtime"
	"unsafe"
)

func GetDefaultNttConfig() core.NTTConfig[*core.Field] {
	cosetGenField := newBN254ScalarField()
	cosetGenField.One()
	return core.GetDefaultNTTConfig(&cosetGenField)
}

// maybe change to CudaErrorT??
func Ntt(scalars cr.HostOrDeviceSlice[any, any], dir core.NTTDir, cfg *core.NTTConfig[*core.Field], results cr.HostOrDeviceSlice[any, any]) core.IcicleError {
	core.NttCheck(scalars, cfg, results)
	cScalars := (*C.scalar_t)(unsafe.Pointer(scalars.AsPointer()))
	cSize := (C.int)(scalars.Len() / int(cfg.BatchSize))
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(unsafe.Pointer(cfg))
	cResults := (*C.scalar_t)(unsafe.Pointer(results.AsPointer()))
	__ret := C.bn254NTTCuda(cScalars, cSize, cDir, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

// TODO: Figure out how to pass primitiveRoot as a value
func InitDomain(primitiveRoot core.FieldInter, ctx cr.DeviceContext) core.IcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(&primitiveRoot.GetLimbs()[0]))
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	__ret := C.bn254InitializeDomainInt(cPrimitiveRoot, cCtx)
	// __ret := 1
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
