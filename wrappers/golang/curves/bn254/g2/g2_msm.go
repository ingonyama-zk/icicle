package g2

// #cgo CFLAGS: -I./include/
// #include "g2_msm.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func G2GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func G2Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError {
	core.MsmCheck(scalars, points, cfg, results)

	scalarsPointer := scalars.AsUnsafePointer()
	cScalars := (*C.scalar_t)(scalarsPointer)

	pointsPointer := points.AsUnsafePointer()
	cPoints := (*C.g2_affine_t)(pointsPointer)

	resultsPointer := results.AsUnsafePointer()
	cResults := (*C.g2_projective_t)(resultsPointer)

	cSize := (C.int)(scalars.Len() / results.Len())
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))

	__ret := C.bn254G2MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}

func G2PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, c int32, ctx *cr.DeviceContext, outputBases core.DeviceSlice) cr.CudaError {
	core.PrecomputeBasesCheck(points, precomputeFactor, outputBases)

	pointsPointer := points.AsUnsafePointer()
	cPoints := (*C.g2_affine_t)(pointsPointer)

	cPointsLen := (C.int)(points.Len())
	cPrecomputeFactor := (C.int)(precomputeFactor)
	cC := (C.int)(c)
	cPointsIsOnDevice := (C._Bool)(points.IsOnDevice())
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))

	outputBasesPointer := outputBases.AsUnsafePointer()
	cOutputBases := (*C.g2_affine_t)(outputBasesPointer)

	__ret := C.bn254G2PrecomputeMSMBases(cPoints, cPointsLen, cPrecomputeFactor, cC, cPointsIsOnDevice, cCtx, cOutputBases)
	err := (cr.CudaError)(__ret)
	return err
}
