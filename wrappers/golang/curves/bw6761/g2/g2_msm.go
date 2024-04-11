package g2

// #cgo CFLAGS: -I./include/
// #include "g2_msm.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	icicle_bw6761 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bw6761"
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
		scalarsPointer = scalarsDevice.AsPointer()
	} else {
		scalarsPointer = unsafe.Pointer(&scalars.(core.HostSlice[icicle_bw6761.ScalarField])[0])
	}
	cScalars := (*C.scalar_t)(scalarsPointer)

	var pointsPointer unsafe.Pointer
	if points.IsOnDevice() {
		pointsDevice := points.(core.DeviceSlice)
		pointsDevice.CheckDevice()
		pointsPointer = pointsDevice.AsPointer()
	} else {
		pointsPointer = unsafe.Pointer(&points.(core.HostSlice[G2Affine])[0])
	}
	cPoints := (*C.g2_affine_t)(pointsPointer)

	var resultsPointer unsafe.Pointer
	if results.IsOnDevice() {
		resultsDevice := results.(core.DeviceSlice)
		resultsDevice.CheckDevice()
		resultsPointer = resultsDevice.AsPointer()
	} else {
		resultsPointer = unsafe.Pointer(&results.(core.HostSlice[G2Projective])[0])
	}
	cResults := (*C.g2_projective_t)(resultsPointer)

	cSize := (C.int)(scalars.Len() / results.Len())
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))

	__ret := C.bw6_761G2MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}

func G2PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, c int32, ctx *cr.DeviceContext, outputBases core.DeviceSlice) cr.CudaError {
	core.PrecomputeBasesCheck(points, precomputeFactor, outputBases)

	var pointsPointer unsafe.Pointer
	if points.IsOnDevice() {
		pointsPointer = points.(core.DeviceSlice).AsPointer()
	} else {
		pointsPointer = unsafe.Pointer(&points.(core.HostSlice[G2Affine])[0])
	}
	cPoints := (*C.g2_affine_t)(pointsPointer)

	cPointsLen := (C.int)(points.Len())
	cPrecomputeFactor := (C.int)(precomputeFactor)
	cC := (C.int)(c)
	cPointsIsOnDevice := (C._Bool)(points.IsOnDevice())
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))

	outputBasesPointer := outputBases.AsPointer()
	cOutputBases := (*C.g2_affine_t)(outputBasesPointer)

	__ret := C.bw6_761G2PrecomputeMSMBases(cPoints, cPointsLen, cPrecomputeFactor, cC, cPointsIsOnDevice, cCtx, cOutputBases)
	err := (cr.CudaError)(__ret)
	return err
}
