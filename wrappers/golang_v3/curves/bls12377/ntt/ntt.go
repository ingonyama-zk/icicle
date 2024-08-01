package ntt

// #cgo CFLAGS: -I./include/
// #include "ntt.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bls12377"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) runtime.EIcicleError {
	scalarsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](scalars, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.scalar_t)(resultsPointer)

	__ret := C.bls12_377_ntt(cScalars, cSize, cDir, cCfg, cResults)
	err := runtime.EIcicleError(__ret)
	return err
}

func GetDefaultNttConfig() core.NTTConfig[[bls12_377.SCALAR_LIMBS]uint32] {
	cosetGenField := bls12_377.ScalarField{}
	cosetGenField.One()
	var cosetGen [bls12_377.SCALAR_LIMBS]uint32
	for i, v := range cosetGenField.GetLimbs() {
		cosetGen[i] = v
	}

	return core.GetDefaultNTTConfig(cosetGen)
}

func GetRootOfUnity(size uint64) bls12_377.ScalarField {
	cRes := C.bls12_377_get_root_of_unity((C.size_t)(size))
	var res bls12_377.ScalarField
	res.FromLimbs(*(*[]uint32)(unsafe.Pointer(cRes)))
	return res
}

func InitDomain(primitiveRoot bls12_377.ScalarField, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(primitiveRoot.AsPointer()))
	cCfg := (*C.NTTInitDomainConfig)(unsafe.Pointer(&cfg))
	__ret := C.bls12_377_ntt_init_domain(cPrimitiveRoot, cCfg)
	err := runtime.EIcicleError(__ret)
	return err
}

func ReleaseDomain() runtime.EIcicleError {
	__ret := C.bls12_377_ntt_release_domain()
	err := runtime.EIcicleError(__ret)
	return err
}
