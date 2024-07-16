package keccak

// #cgo CFLAGS: -I./include/
// #include "keccak.h"
import "C"

import (
	"fmt"
	"unsafe"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

type HashSize int

const (
	Hash256 HashSize = 256
	Hash512 HashSize = 512
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

func keccakCheck(input core.HostOrDeviceSlice, output core.HostOrDeviceSlice, cfg *KeccakConfig, hashSize HashSize, numberOfBlocks int32) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer) {
	cfg.areInputsOnDevice = input.IsOnDevice()
	cfg.areOutputsOnDevice = output.IsOnDevice()

	if input.IsOnDevice() {
		input.(core.DeviceSlice).CheckDevice()
	}

	if output.IsOnDevice() {
		output.(core.DeviceSlice).CheckDevice()
	}

	if output.Cap() < int(hashSize)/8*int(numberOfBlocks) {
		errorString := fmt.Sprintf(
			"Output capacity %d isn't enough for hashSize %d and numberOfBlocks %d",
			output.Cap(),
			hashSize,
			numberOfBlocks,
		)
		panic(errorString)
	}

	return input.AsUnsafePointer(), output.AsUnsafePointer(), unsafe.Pointer(cfg)
}

func keccak(input core.HostOrDeviceSlice, inputBlockSize, numberOfBlocks int32, output core.HostOrDeviceSlice, config *KeccakConfig, hashSize HashSize) (ret core.IcicleError) {
	inputPointer, outputPointer, cfgPointer := keccakCheck(input, output, config, hashSize, numberOfBlocks)
	cInput := (*C.uint8_t)(inputPointer)
	cOutput := (*C.uint8_t)(outputPointer)
	cInputBlockSize := (C.int)(inputBlockSize)
	cNumberOfBlocks := (C.int)(numberOfBlocks)
	cConfig := (*C.KeccakConfig)(cfgPointer)

	switch hashSize {
	case Hash256:
		ret = core.FromCudaError((cr.CudaError)(C.keccak256_cuda(cInput, cInputBlockSize, cNumberOfBlocks, cOutput, cConfig)))
	case Hash512:
		ret = core.FromCudaError((cr.CudaError)(C.keccak512_cuda(cInput, cInputBlockSize, cNumberOfBlocks, cOutput, cConfig)))
	}

	return ret
}

func Keccak256(input core.HostOrDeviceSlice, inputBlockSize, numberOfBlocks int32, output core.HostOrDeviceSlice, config *KeccakConfig) core.IcicleError {
	return keccak(input, inputBlockSize, numberOfBlocks, output, config, Hash256)
}

func Keccak512(input core.HostOrDeviceSlice, inputBlockSize, numberOfBlocks int32, output core.HostOrDeviceSlice, config *KeccakConfig) core.IcicleError {
	return keccak(input, inputBlockSize, numberOfBlocks, output, config, Hash512)
}
