package ecntt

// #cgo CFLAGS: -I./include/
// #include "ecntt.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

func ECNtt[T any](points core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) core.IcicleError {
	pointsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](points, cfg, results)

	cPoints := (*C.projective_t)(pointsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.projective_t)(resultsPointer)

	__ret := C.bw6_761_ecntt_cuda(cPoints, cSize, cDir, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
