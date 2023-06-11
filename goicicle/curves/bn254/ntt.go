package bn254

// #cgo CFLAGS: -I../../../icicle/curves/bn254/
// #cgo LDFLAGS: -L../../../icicle/curves/bn254/ -lbn254
// #include "ntt.h"
import "C"
import "unsafe"

func NttBN254(scalars *[]ScalarField, isInverse bool, deviceId int) uint64 {
	scalarsC := (*C.BN254_scalar_t)(unsafe.Pointer(&(*scalars)[0]))

	ret := C.ntt_cuda_bn254(scalarsC, C.uint32_t(len(*scalars)), C.bool(isInverse), C.size_t(deviceId))

	return uint64(ret)
}

func NttBatchBN254(scalars *[]ScalarField, isInverse bool, batchSize, deviceId int) uint64 {
	scalarsC := (*C.BN254_scalar_t)(unsafe.Pointer(&(*scalars)[0]))
	isInverseC := C.bool(isInverse)
	batchSizeC := C.uint32_t(batchSize)
	deviceIdC := C.size_t(deviceId)

	ret := C.ntt_batch_cuda_bn254(scalarsC, C.uint32_t(len(*scalars)), batchSizeC, isInverseC, deviceIdC)

	return uint64(ret)
}

func EcNttBN254(values *[]PointBN254, isInverse bool, deviceId int) uint64 {
	valuesC := (*C.BN254_projective_t)(unsafe.Pointer(&(*values)[0]))
	deviceIdC := C.size_t(deviceId)
	isInverseC := C.bool(isInverse)
	n := C.uint32_t(len(*values))

	ret := C.ecntt_cuda_bn254(valuesC, n, isInverseC, deviceIdC)

	return uint64(ret)
}

func EcNttBatchBN254(values *[]PointBN254, isInverse bool, batchSize, deviceId int) uint64 {
	valuesC := (*C.BN254_projective_t)(unsafe.Pointer(&(*values)[0]))
	deviceIdC := C.size_t(deviceId)
	isInverseC := C.bool(isInverse)
	n := C.uint32_t(len(*values))
	batchSizeC := C.uint32_t(batchSize)

	ret := C.ecntt_batch_cuda_bn254(valuesC, n, batchSizeC, isInverseC, deviceIdC)

	return uint64(ret)
}
