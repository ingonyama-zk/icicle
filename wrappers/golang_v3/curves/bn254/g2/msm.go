package g2

// // #cgo CFLAGS: -I./include/
// // #include "msm.h"
// import "C"

// import (
// 	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
// 	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
// 	"unsafe"
// )

// func G2GetDefaultMSMConfig() core.MSMConfig {
// 	return core.GetDefaultMSMConfig()
// }

// func G2Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError {
// 	scalarsPointer, pointsPointer, resultsPointer, size, cfgPointer := core.MsmCheck(scalars, points, cfg, results)

// 	cScalars := (*C.scalar_t)(scalarsPointer)
// 	cPoints := (*C.g2_affine_t)(pointsPointer)
// 	cResults := (*C.g2_projective_t)(resultsPointer)
// 	cSize := (C.int)(size)
// 	cCfg := (*C.MSMConfig)(cfgPointer)

// 	__ret := C.bn254_g2_msm_cuda(cScalars, cPoints, cSize, cCfg, cResults)
// 	err := (cr.CudaError)(__ret)
// 	return err
// }

// // Deprecated: G2PrecomputeBases exists for backward compatibility.
// // It may cause issues if an MSM with a different `c` value is used with precomputed points and it will be removed in a future version.
// // G2PrecomputePoints should be used instead.
// func G2PrecomputeBases(points core.HostOrDeviceSlice, precomputeFactor int32, c int32, ctx *cr.DeviceContext, outputBases core.DeviceSlice) cr.CudaError {
// 	cfg := G2GetDefaultMSMConfig()
// 	cfg.PrecomputeFactor = precomputeFactor
// 	pointsPointer, outputBasesPointer := core.PrecomputePointsCheck(points, &cfg, outputBases)

// 	cPoints := (*C.g2_affine_t)(pointsPointer)
// 	cPointsLen := (C.int)(points.Len())
// 	cPrecomputeFactor := (C.int)(precomputeFactor)
// 	cC := (C.int)(c)
// 	cPointsIsOnDevice := (C._Bool)(points.IsOnDevice())
// 	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))
// 	cOutputBases := (*C.g2_affine_t)(outputBasesPointer)

// 	__ret := C.bn254_g2_precompute_msm_bases_cuda(cPoints, cPointsLen, cPrecomputeFactor, cC, cPointsIsOnDevice, cCtx, cOutputBases)
// 	err := (cr.CudaError)(__ret)
// 	return err
// }

// func G2PrecomputePoints(points core.HostOrDeviceSlice, msmSize int, cfg *core.MSMConfig, outputBases core.DeviceSlice) cr.CudaError {
// 	pointsPointer, outputBasesPointer := core.PrecomputePointsCheck(points, cfg, outputBases)

// 	cPoints := (*C.g2_affine_t)(pointsPointer)
// 	cMsmSize := (C.int)(msmSize)
// 	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))
// 	cOutputBases := (*C.g2_affine_t)(outputBasesPointer)

// 	__ret := C.bn254_g2_precompute_msm_points_cuda(cPoints, cMsmSize, cCfg, cOutputBases)
// 	err := (cr.CudaError)(__ret)
// 	return err
// }
