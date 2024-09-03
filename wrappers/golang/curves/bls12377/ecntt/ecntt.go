package ecntt

// #cgo CFLAGS: -I./include/
// #include "ecntt.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func ECNtt[T any](points core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) runtime.EIcicleError {
	pointsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](points, cfg, results)

	cPoints := (*C.projective_t)(pointsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.projective_t)(resultsPointer)

	__ret := C.bls12_377_ecntt(cPoints, cSize, cDir, cCfg, cResults)
	err := runtime.EIcicleError(__ret)
	return err
}
