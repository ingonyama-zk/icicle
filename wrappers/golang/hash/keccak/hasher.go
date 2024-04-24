package keccak

// #cgo CFLAGS: -I./include/
// #include "keccak.h"
import "C"

import (
	"unsafe"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

type KeccakConfig struct {
	Ctx                cr.DeviceContext
	areInputsOnDevice  bool
	areOutputsOnDevice bool
	IsAsync            bool
}

func GetDefaultKeccakConfig() KeccakConfig {
	ctx, _ := cr.GetDefaultDeviceContext()
	return KeccakConfig{
		ctx,
		false,
		false,
		false,
	}
}

func KeccakCheck(input core.HostOrDeviceSlice, output core.HostOrDeviceSlice, cfg *KeccakConfig) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer) {
	cfg.areInputsOnDevice = input.IsOnDevice()
	cfg.areOutputsOnDevice = output.IsOnDevice()

	if input.IsOnDevice() {
		input.(core.DeviceSlice).CheckDevice()
	}

	if output.IsOnDevice() {
		output.(core.DeviceSlice).CheckDevice()
	}

	return input.AsUnsafePointer(), output.AsUnsafePointer(), unsafe.Pointer(cfg)
}

func Keccak256(input core.HostOrDeviceSlice, inputBlockSize, numberOfBlocks int32, output core.HostOrDeviceSlice, config *KeccakConfig) core.IcicleError {
	inputPointer, outputPointer, cfgPointer := KeccakCheck(input, output, config)
	cInput := (*C.uint8_t)(inputPointer)
	cOutput := (*C.uint8_t)(outputPointer)
	cInputBlockSize := (C.int)(inputBlockSize)
	cNumberOfBlocks := (C.int)(numberOfBlocks)
	cConfig := (*C.KeccakConfig)(cfgPointer)

	__ret := C.keccak256_cuda(cInput, cInputBlockSize, cNumberOfBlocks, cOutput, cConfig)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func Keccak512(input core.HostOrDeviceSlice, inputBlockSize, numberOfBlocks int32, output core.HostOrDeviceSlice, config *KeccakConfig) core.IcicleError {
	inputPointer, outputPointer, cfgPointer := KeccakCheck(input, output, config)
	cInput := (*C.uint8_t)(inputPointer)
	cOutput := (*C.uint8_t)(outputPointer)
	cInputBlockSize := (C.int)(inputBlockSize)
	cNumberOfBlocks := (C.int)(numberOfBlocks)
	cConfig := (*C.KeccakConfig)(cfgPointer)

	__ret := C.keccak512_cuda(cInput, cInputBlockSize, cNumberOfBlocks, cOutput, cConfig)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
