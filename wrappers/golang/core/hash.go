package core

import (
	"fmt"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
)

type HashConfig struct {
	StreamHandle       runtime.Stream
	batchSize          uint64
	areInputsOnDevice  bool
	areOutputsOnDevice bool
	IsAsync            bool
	Ext                config_extension.ConfigExtensionHandler
}

func GetDefaultHashConfig() HashConfig {
	return HashConfig{
		StreamHandle:       nil,
		batchSize:          1,
		areInputsOnDevice:  false,
		areOutputsOnDevice: false,
		IsAsync:            false,
		Ext:                nil,
	}
}

func HashCheck(input, output HostOrDeviceSlice, outputSize uint64, cfg *HashConfig) (unsafe.Pointer, unsafe.Pointer, uint64, runtime.EIcicleError) {
	var inputByteSize, outputByteSize uint64

	if input.IsOnDevice() {
		input.(DeviceSlice).CheckDevice()
		cfg.areInputsOnDevice = true
		inputByteSize = uint64(input.(DeviceSlice).Cap())
	} else {
		inputByteSize = uint64(input.(HostSlice[byte]).Cap())
	}

	if output.IsOnDevice() {
		output.(DeviceSlice).CheckDevice()
		cfg.areOutputsOnDevice = true
		outputByteSize = uint64(output.(DeviceSlice).Cap())
	} else {
		outputByteSize = uint64(output.(HostSlice[byte]).SizeOfElement() * output.Len())
	}

	if outputByteSize%outputSize != 0 {
		errorString := fmt.Sprintf("output size (%d Bytes) must divide the output size of a single hash (%d Bytes)", outputByteSize, outputSize)
		fmt.Println(errorString)
		return nil, nil, 0, runtime.InvalidArgument
	}

	cfg.batchSize = outputByteSize / outputSize

	if inputByteSize%cfg.batchSize != 0 {
		errorString := fmt.Sprintf("input size (%d Bytes) must divide batch size (%d; batchSize = outputByteSize / outputSize)", inputByteSize, cfg.batchSize)
		fmt.Println(errorString)
		return nil, nil, 0, runtime.InvalidArgument
	}

	lengthPerBatch := inputByteSize / cfg.batchSize

	return input.AsUnsafePointer(), output.AsUnsafePointer(), lengthPerBatch, runtime.Success
}
