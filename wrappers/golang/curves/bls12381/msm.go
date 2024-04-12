package bls12381

// #cgo CFLAGS: -I./include/
// #include "msm.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError {
	core.MsmCheck(scalars, points, cfg, results)

	scalarsPointer := scalars.GetPointerSafe()
	cScalars := (*C.scalar_t)(scalarsPointer)

	pointsPointer := points.GetPointerSafe()
	cPoints := (*C.affine_t)(pointsPointer)

	resultsPointer := results.GetPointerSafe()
	cResults := (*C.projective_t)(resultsPointer)

	cSize := (C.int)(scalars.Len() / results.Len())
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))

	__ret := C.bls12_381MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}

func PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, c int32, ctx *cr.DeviceContext, outputBases core.DeviceSlice) cr.CudaError {
	core.PrecomputeBasesCheck(points, precomputeFactor, outputBases)

	pointsPointer := points.GetPointerSafe()
	cPoints := (*C.affine_t)(pointsPointer)

	cPointsLen := (C.int)(points.Len())
	cPrecomputeFactor := (C.int)(precomputeFactor)
	cC := (C.int)(c)
	cPointsIsOnDevice := (C._Bool)(points.IsOnDevice())
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))

	outputBasesPointer := outputBases.AsPointer()
	cOutputBases := (*C.affine_t)(outputBasesPointer)

	__ret := C.bls12_381PrecomputeMSMBases(cPoints, cPointsLen, cPrecomputeFactor, cC, cPointsIsOnDevice, cCtx, cOutputBases)
	err := (cr.CudaError)(__ret)
	return err
}
