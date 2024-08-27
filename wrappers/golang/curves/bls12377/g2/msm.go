package g2

// #cgo CFLAGS: -I./include/
// #include "msm.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func G2GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func G2Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) runtime.EIcicleError {
	scalarsPointer, pointsPointer, resultsPointer, size := core.MsmCheck(scalars, points, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cPoints := (*C.g2_affine_t)(pointsPointer)
	cResults := (*C.g2_projective_t)(resultsPointer)
	cSize := (C.int)(size)
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))

	__ret := C.bls12_377_g2_msm(cScalars, cPoints, cSize, cCfg, cResults)
	err := runtime.EIcicleError(__ret)
	return err
}

func G2PrecomputeBases(bases core.HostOrDeviceSlice, cfg *core.MSMConfig, outputBases core.DeviceSlice) runtime.EIcicleError {
	basesPointer, outputBasesPointer := core.PrecomputeBasesCheck(bases, cfg, outputBases)

	cBases := (*C.g2_affine_t)(basesPointer)
	var cBasesLen C.int
	if cfg.AreBasesShared {
		cBasesLen = (C.int)(bases.Len())
	} else {
		cBasesLen = (C.int)(bases.Len() / int(cfg.BatchSize))
	}
	cCfg := (*C.MSMConfig)(unsafe.Pointer(cfg))
	cOutputBases := (*C.g2_affine_t)(outputBasesPointer)

	__ret := C.bls12_377_g2_msm_precompute_bases(cBases, cBasesLen, cCfg, cOutputBases)
	err := runtime.EIcicleError(__ret)
	return err
}
