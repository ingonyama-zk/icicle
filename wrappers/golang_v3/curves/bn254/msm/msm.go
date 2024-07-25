package msm

// #cgo CFLAGS: -I./include/
// #include "msm.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

func GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) runtime.EIcicleError {
	scalarsPointer, pointsPointer, resultsPointer, size := core.MsmCheck(scalars, points, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cPoints := (*C.affine_t)(pointsPointer)
	cResults := (*C.projective_t)(resultsPointer)
	cSize := (C.int)(size)
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))

	__ret := C.bn254_msm(cScalars, cPoints, cSize, cCfg, cResults)
	err := runtime.EIcicleError(__ret)
	return err
}

func PrecomputeBases(bases core.HostOrDeviceSlice, cfg *core.MSMConfig, outputBases core.DeviceSlice) runtime.EIcicleError {
	basesPointer, outputBasesPointer := core.PrecomputeBasesCheck(bases, cfg, outputBases)

	cBases := (*C.affine_t)(basesPointer)
	var cBasesLen C.int
	if cfg.AreBasesShared {
		cBasesLen = (C.int)(bases.Len())
	} else {
		cBasesLen = (C.int)(bases.Len() / int(cfg.BatchSize))
	}
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))
	cOutputBases := (*C.affine_t)(outputBasesPointer)

	__ret := C.bn254_msm_precompute_bases(cBases, cBasesLen, cCfg, cOutputBases)
	err := runtime.EIcicleError(__ret)
	return err
}
