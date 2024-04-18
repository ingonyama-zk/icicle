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

	var scalarsPointer unsafe.Pointer
	if scalars.IsOnDevice() {
		scalarsDevice := scalars.(core.DeviceSlice)
		scalarsDevice.CheckDevice()
		scalarsPointer = scalarsDevice.AsUnsafePointer()
	} else {
		scalarsPointer = scalars.AsUnsafePointer()
	}
	cScalars := (*C.scalar_t)(scalarsPointer)

	var pointsPointer unsafe.Pointer
	if points.IsOnDevice() {
		pointsDevice := points.(core.DeviceSlice)
		pointsDevice.CheckDevice()
		pointsPointer = pointsDevice.AsUnsafePointer()
	} else {
		pointsPointer = points.AsUnsafePointer()
	}
	cPoints := (*C.g2_affine_t)(pointsPointer)

	var resultsPointer unsafe.Pointer
	if results.IsOnDevice() {
		resultsDevice := results.(core.DeviceSlice)
		resultsDevice.CheckDevice()
		resultsPointer = resultsDevice.AsUnsafePointer()
	} else {
		resultsPointer = results.AsUnsafePointer()
	}
	cResults := (*C.g2_projective_t)(resultsPointer)

	cSize := (C.int)(scalars.Len() / results.Len())
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))

	__ret := C.bls12_377G2MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}

func G2PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, c int32, ctx *cr.DeviceContext, outputBases core.DeviceSlice) cr.CudaError {
	core.PrecomputeBasesCheck(points, precomputeFactor, outputBases)

	var pointsPointer unsafe.Pointer
	if points.IsOnDevice() {
		pointsDevice := points.(core.DeviceSlice)
		pointsDevice.CheckDevice()
		pointsPointer = pointsDevice.AsUnsafePointer()
	} else {
		pointsPointer = points.AsUnsafePointer()
	}
	cPoints := (*C.g2_affine_t)(pointsPointer)

	cPointsLen := (C.int)(points.Len())
	cPrecomputeFactor := (C.int)(precomputeFactor)
	cC := (C.int)(c)
	cPointsIsOnDevice := (C._Bool)(points.IsOnDevice())
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))

	outputBasesPointer := outputBases.AsUnsafePointer()
	cOutputBases := (*C.g2_affine_t)(outputBasesPointer)

	__ret := C.bls12_377G2PrecomputeMSMBases(cPoints, cPointsLen, cPrecomputeFactor, cC, cPointsIsOnDevice, cCtx, cOutputBases)
	err := (cr.CudaError)(__ret)
	return err
}
