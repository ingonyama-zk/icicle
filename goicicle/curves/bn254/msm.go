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
		return nil, fmt.Errorf("msm_cuda_bn254 returned error code: %d", ret)
	}

	return out, nil
}

func MsmBatchBN254(points *[]PointAffineNoInfinityBN254, scalars *[]FieldBN254, batchSize, deviceId int) ([]*PointBN254, error) {
	// Check for nil pointers
	if points == nil || scalars == nil {
		return nil, errors.New("points or scalars is nil")
	}

	if len(*points) != len(*scalars) {
		return nil, errors.New("error on: len(points) != len(scalars)")
	}

	// Check for empty slices
	if len(*points) == 0 || len(*scalars) == 0 {
		return nil, errors.New("points or scalars is empty")
	}

	// Check for zero batchSize
	if batchSize <= 0 {
		return nil, errors.New("error on: batchSize must be greater than zero")
	}

	out := make([]*PointBN254, batchSize)

	for i := 0; i < len(out); i++ {
		out[i] = NewPointBN254Zero()
	}

	outC := (*C.BN254_projective_t)(unsafe.Pointer(&out[0]))
	pointsC := (*C.BN254_affine_t)(unsafe.Pointer(&(*points)[0]))
	scalarsC := (*C.BN254_scalar_t)(unsafe.Pointer(&(*scalars)[0]))
	msmSizeC := C.size_t(len(*points) / batchSize)
	deviceIdC := C.size_t(deviceId)
	batchSizeC := C.size_t(batchSize)

	ret := C.msm_batch_cuda_bn254(outC, pointsC, scalarsC, batchSizeC, msmSizeC, deviceIdC)
	if ret != 0 {
		return nil, fmt.Errorf("msm_batch_cuda_bn254 returned error code: %d", ret)
	}

	return out, nil
}
