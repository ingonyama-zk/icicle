//go:build !icicle_exclude_all || poseidon2

package poseidon2

// #cgo CFLAGS: -I./include/
// #include "poseidon2.h"
import "C"

import (
	"unsafe"

	bls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func NewHasher(t uint64, domainTag *bls12_381.ScalarField) (hash.Hasher, runtime.EIcicleError) {
	cT := (C.uint)(t)
	var cDomainTag *C.scalar_t // This is set to nil as zero value
	if domainTag != nil {
		cDomainTag = (*C.scalar_t)(unsafe.Pointer(domainTag.AsPointer()))
	}

	handle := C.bls12_381_create_poseidon2_hasher(cT, cDomainTag, 0)

	if handle == nil {
		return hash.Hasher{}, runtime.UnknownError
	}

	return hash.FromHandle(unsafe.Pointer(handle)), runtime.Success
}
