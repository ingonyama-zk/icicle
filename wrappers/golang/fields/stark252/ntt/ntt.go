//go:build !ntt

package ntt

// #cgo CFLAGS: -I./include/
// #include "ntt.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	stark252 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/stark252"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) runtime.EIcicleError {
	scalarsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](scalars, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.scalar_t)(resultsPointer)

	__ret := C.stark252_ntt(cScalars, cSize, cDir, cCfg, cResults)
	err := runtime.EIcicleError(__ret)
	return err
}

func GetDefaultNttConfig() core.NTTConfig[[stark252.SCALAR_LIMBS]uint32] {
	cosetGenField := stark252.ScalarField{}
	cosetGenField.One()
	var cosetGen [stark252.SCALAR_LIMBS]uint32
	for i, v := range cosetGenField.GetLimbs() {
		cosetGen[i] = v
	}

	return core.GetDefaultNTTConfig(cosetGen)
}

func GetRootOfUnity(size uint64) stark252.ScalarField {
	var res stark252.ScalarField
	cErr := C.stark252_get_root_of_unity((C.size_t)(size), (*C.scalar_t)(unsafe.Pointer(&res)))
	if runtime.EIcicleError(cErr) != runtime.Success {
		panic("Failed to get root of unity")
	}
	return res
}

func InitDomain(primitiveRoot stark252.ScalarField, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(primitiveRoot.AsPointer()))
	cCfg := (*C.NTTInitDomainConfig)(unsafe.Pointer(&cfg))
	__ret := C.stark252_ntt_init_domain(cPrimitiveRoot, cCfg)
	err := runtime.EIcicleError(__ret)
	return err
}

func ReleaseDomain() runtime.EIcicleError {
	__ret := C.stark252_ntt_release_domain()
	err := runtime.EIcicleError(__ret)
	return err
}
