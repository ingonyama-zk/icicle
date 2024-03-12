package bls12377

// #cgo CFLAGS: -I./include/
// #include "msm.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	"unsafe"
)

func GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError {
	core.MsmCheck(scalars, points, cfg, results)
	var scalarsPointer unsafe.Pointer
	if scalars.IsOnDevice() {
		scalarsPointer = scalars.(core.DeviceSlice).AsPointer()
	} else {
		scalarsPointer = unsafe.Pointer(&scalars.(core.HostSlice[ScalarField])[0])
	}
	cScalars := (*C.scalar_t)(scalarsPointer)

	var pointsPointer unsafe.Pointer
	if points.IsOnDevice() {
		pointsPointer = points.(core.DeviceSlice).AsPointer()
	} else {
		pointsPointer = unsafe.Pointer(&points.(core.HostSlice[Affine])[0])
	}
	cPoints := (*C.affine_t)(pointsPointer)

	var resultsPointer unsafe.Pointer
	if results.IsOnDevice() {
		resultsPointer = results.(core.DeviceSlice).AsPointer()
	} else {
		resultsPointer = unsafe.Pointer(&results.(core.HostSlice[Projective])[0])
	}
	cResults := (*C.projective_t)(resultsPointer)

	cSize := (C.int)(scalars.Len() / results.Len())
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))

	__ret := C.bls12_377MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}

func PrecomputeBases(points core.HostOrDeviceSlice, precompute_factor int32, _c int32, ctx *cr.DeviceContext, output_bases core.HostOrDeviceSlice) cr.CudaError {
	core.PrecomputeBasesCheck(points, precompute_factor, output_bases)

	var pointsPointer unsafe.Pointer
	if points.IsOnDevice() {
		pointsPointer = points.(core.DeviceSlice).AsPointer()
	} else {
		pointsPointer = unsafe.Pointer(&points.(core.HostSlice[Affine])[0])
	}
	cPoints := (*C.affine_t)(pointsPointer)

	cPointsLen := (C.int)(points.Len())
	cPrecomputeFactor := (C.int)(precompute_factor)
	c_C := (C.int)(_c)
	cPointsIsOnDevice := (C._Bool)(points.IsOnDevice())
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))

	var outputBasesPointer unsafe.Pointer
	if output_bases.IsOnDevice() {
		outputBasesPointer = output_bases.(core.DeviceSlice).AsPointer()
	} else {
		outputBasesPointer = unsafe.Pointer(&output_bases.(core.HostSlice[Projective])[0])
	}
	cOutputBases := (*C.affine_t)(outputBasesPointer)

	__ret := C.bls12_377PrecomputeMSMBases(cPoints, cPointsLen, cPrecomputeFactor, c_C, cPointsIsOnDevice, cCtx, cOutputBases)
	err := (cr.CudaError)(__ret)
	return err
}
