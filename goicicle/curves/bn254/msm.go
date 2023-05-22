package bn254

import (
	"errors"
	"fmt"
	"unsafe"
)

// #cgo CFLAGS: -I../../../icicle/curves/bn254/
// #cgo LDFLAGS: -L../../../icicle/curves/bn254/ -lbn254
// #include "msm.h"
import "C"

func MsmBN254(points []PointAffineNoInfinityBN254, scalars []FieldBN254, device_id int) (*PointBN254, error) {
	if len(points) != len(scalars) {
		return nil, errors.New("error on: len(points) != len(scalars)")
	}

	out := new(PointBN254)

	pointsC := (*C.BN254_affine_t)(unsafe.Pointer(&points[0]))
	scalarsC := (*C.BN254_scalar_t)(unsafe.Pointer(&scalars[0]))
	outC := (*C.BN254_projective_t)(unsafe.Pointer(out))

	ret := C.msm_cuda_bn254(outC, pointsC, scalarsC, C.size_t(len(points)), C.size_t(device_id))

	if ret != 0 {
		return nil, fmt.Errorf("msm_cuda_bn254_wrapper returned error code: %d", ret)
	}

	return out, nil
}
