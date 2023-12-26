package bn254

// #cgo CFLAGS: -I./include/
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254
// #include "msm.h"
import "C"

import (
	"unsafe"
	"local/hello/icicle/wrappers/golang/core"
	cr "local/hello/icicle/wrappers/golang/cuda_runtime"
)

func GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

// maybe change to CudaErrorT??
func Msm(scalars cr.HostOrDeviceSlice[any, any],	points cr.HostOrDeviceSlice[any, any], cfg *core.MSMConfig, results cr.HostOrDeviceSlice[any, any]) cr.CudaError {
	core.MsmCheck(scalars, points, cfg, results)
	cScalars := (*C.scalar_t)(unsafe.Pointer(scalars.AsPointer()))
	cPoints := (*C.affine_t)(unsafe.Pointer(points.AsPointer()))
	cSize := (C.int)(scalars.Len() / results.Len())
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))
	cResults := (*C.projective_t)(unsafe.Pointer(results.AsPointer()))
	__ret := C.bn254MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}
