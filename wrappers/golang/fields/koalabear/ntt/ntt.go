package ntt

// #cgo CFLAGS: -I./include/
// #include "ntt.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	koalabear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func Ntt[T any](scalars core.HostOrDeviceSlice, dir core.NTTDir, cfg *core.NTTConfig[T], results core.HostOrDeviceSlice) runtime.EIcicleError {
	scalarsPointer, resultsPointer, size, cfgPointer := core.NttCheck[T](scalars, cfg, results)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cSize := (C.int)(size)
	cDir := (C.int)(dir)
	cCfg := (*C.NTTConfig)(cfgPointer)
	cResults := (*C.scalar_t)(resultsPointer)

	__ret := C.koalabear_ntt(cScalars, cSize, cDir, cCfg, cResults)
	err := runtime.EIcicleError(__ret)
	return err
}

func GetDefaultNttConfig() core.NTTConfig[[koalabear.SCALAR_LIMBS]uint32] {
	cosetGenField := koalabear.ScalarField{}
	cosetGenField.One()
	var cosetGen [koalabear.SCALAR_LIMBS]uint32
	for i, v := range cosetGenField.GetLimbs() {
		cosetGen[i] = v
	}

	return core.GetDefaultNTTConfig(cosetGen)
}

func GetRootOfUnity(size uint64) koalabear.ScalarField {
	cRes := C.koalabear_get_root_of_unity((C.size_t)(size))
	var res koalabear.ScalarField
	res.FromLimbs(*(*[]uint32)(unsafe.Pointer(cRes)))
	return res
}

func InitDomain(primitiveRoot koalabear.ScalarField, cfg core.NTTInitDomainConfig) runtime.EIcicleError {
	cPrimitiveRoot := (*C.scalar_t)(unsafe.Pointer(primitiveRoot.AsPointer()))
	cCfg := (*C.NTTInitDomainConfig)(unsafe.Pointer(&cfg))
	__ret := C.koalabear_ntt_init_domain(cPrimitiveRoot, cCfg)
	err := runtime.EIcicleError(__ret)
	return err
}

func ReleaseDomain() runtime.EIcicleError {
	__ret := C.koalabear_ntt_release_domain()
	err := runtime.EIcicleError(__ret)
	return err
}
