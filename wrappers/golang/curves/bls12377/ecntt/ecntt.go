package ecntt

// #cgo CFLAGS: -I./include/
// #include "ecntt.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func ECNtt[T any](points core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) core.IcicleError {
	core.NttCheck(points, cfg, results)

	pointsPointer := points.GetPointerSafe()
	cPoints := (*C.projective_t)(pointsPointer)

	cSize := (C.int)(points.Len() / int(cfg.BatchSize))
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(unsafe.Pointer(cfg))

	resultsPointer := results.GetPointerSafe()
	cResults := (*C.projective_t)(resultsPointer)

	__ret := C.bls12_377ECNTTCuda(cPoints, cSize, cDir, cCfg, cResults)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
