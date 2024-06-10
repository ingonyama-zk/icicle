package msm

// #cgo CFLAGS: -I./include/
// #include "msm.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	"unsafe"
)

func GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError {
	scalarsPointer, pointsPointer, resultsPointer, size, cfgPointer := core.MsmCheck(scalars, points, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cPoints := (*C.affine_t)(pointsPointer)
	cResults := (*C.projective_t)(resultsPointer)
	cSize := (C.int)(size)
	cCfg := (*C.MSMConfig)(cfgPointer)

	__ret := C.bn254_msm_cuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}

func PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, msmSize int, cfg *core.MSMConfig, outputBases core.DeviceSlice) cr.CudaError {
	pointsPointer, outputBasesPointer := core.PrecomputeBasesCheck(points, precomputeFactor, outputBases)

	cPoints := (*C.affine_t)(pointsPointer)
	cMsmSize := (C.int)(msmSize)
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))
	cOutputBases := (*C.affine_t)(outputBasesPointer)

	__ret := C.bn254_precompute_msm_bases_cuda(cPoints, cMsmSize, cCfg, cOutputBases)
	err := (cr.CudaError)(__ret)
	return err
}
