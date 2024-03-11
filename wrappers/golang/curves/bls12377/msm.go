package bls12377

// #cgo CFLAGS: -I./include/
// #include "msm.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError {
	scalarsP, pointsP, cfgP, size, resultsP := core.MsmCheck[ScalarField, Affine, Projective](scalars, points, cfg, results)
	cScalars := (*C.scalar_t)(scalarsP)
	cPoints := (*C.affine_t)(pointsP)
	cCfg := (*C.MSMConfig)(cfgP)
	cSize := (C.int)(size)
	cResults := (*C.projective_t)(resultsP)

	__ret := C.bls12_377MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}
