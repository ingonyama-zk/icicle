//go:build !icicle_exclude_all || ntt

package ntt

// #cgo CFLAGS: -I./include/
// #include "ntt.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	goldilocks "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/goldilocks"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) runtime.EIcicleError {
	scalarsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](scalars, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.scalar_t)(resultsPointer)

	__ret := C.goldilocks_ntt(cScalars, cSize, cDir, cCfg, cResults)
	err := runtime.EIcicleError(__ret)
	return err
}

func GetDefaultNttConfig() core.NTTConfig[[goldilocks.SCALAR_LIMBS]uint32] {
	cosetGenField := goldilocks.ScalarField{}
	cosetGenField.One()
	var cosetGen [goldilocks.SCALAR_LIMBS]uint32
	for i, v := range cosetGenField.GetLimbs() {
		cosetGen[i] = v
	}

	return core.GetDefaultNTTConfig(cosetGen)
}

func GetRootOfUnity(size uint64) goldilocks.ScalarField {
	var res goldilocks.ScalarField
	cErr := C.goldilocks_get_root_of_unity((C.size_t)(size), (*C.scalar_t)(unsafe.Pointer(&res)))
	if runtime.EIcicleError(cErr) != runtime.Success {
		panic("Failed to get root of unity")
	}
	return res
}

func InitDomain(primitiveRoot goldilocks.ScalarField, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(primitiveRoot.AsPointer()))
	cCfg := (*C.NTTInitDomainConfig)(unsafe.Pointer(&cfg))
	__ret := C.goldilocks_ntt_init_domain(cPrimitiveRoot, cCfg)
	err := runtime.EIcicleError(__ret)
	return err
}

func ReleaseDomain() runtime.EIcicleError {
	__ret := C.goldilocks_ntt_release_domain()
	err := runtime.EIcicleError(__ret)
	return err
}
