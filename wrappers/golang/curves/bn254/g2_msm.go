//go:build g2

package bn254

// #cgo CFLAGS: -I./include/
// #include "g2_msm.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func G2GetDefaultMSMConfig() core.MSMConfig {
	return core.GetDefaultMSMConfig()
}

func G2Msm(scalars core.HostOrDeviceSlice, points core.HostOrDeviceSlice, cfg *core.MSMConfig, results core.HostOrDeviceSlice) cr.CudaError {
	scalarsP, pointsP, cfgP, size, resultsP := core.MsmCheck[ScalarField, G2Affine, G2Projective](scalars, points, cfg, results)
	cScalars := (*C.scalar_t)(scalarsP)
	cPoints := (*C.g2_affine_t)(pointsP)
	cCfg := (*C.MSMConfig)(cfgP)
	cSize := (C.int)(size)
	cResults := (*C.g2_projective_t)(resultsP)

	__ret := C.bn254G2MSMCuda(cScalars, cPoints, cSize, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return err
}
