package core

import (
	"fmt"
	"local/hello/icicle/wrappers/golang/cuda_runtime"
)

type NTTDir int8

const (
	KForward	NTTDir = iota
	KInverse	NTTDir = 1
)

type Ordering int8

const (
	KNN		Ordering = iota
	KNR		Ordering = 1
	KRN		Ordering = 2
	KRR		Ordering = 3
)

type NTTConfig[S FieldInter] struct {
	/// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
	Ctx cuda_runtime.DeviceContext
	/// Coset generator. Used to perform coset (i)NTTs. Default value: `S::one()` (corresponding to no coset being used).
	CosetGen S
	/// The number of NTTs to compute. Default value: 1.
	BatchSize int32
	/// Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`.
	Ordering Ordering
	areInputsOnDevice bool
	areOutputsOnDevice bool
	/// Whether to run the NTT asynchronously. If set to `true`, the NTT function will be non-blocking and you'd need to synchronize
	/// it explicitly by running `stream.synchronize()`. If set to false, the NTT function will block the current CPU thread.
	IsAsync bool
}

func GetDefaultNTTConfig[S FieldInter](cosetGenOne S) NTTConfig[S] {
	ctx, _ := cuda_runtime.GetDefaultDeviceContext()
	return NTTConfig[S]{
			ctx,   				// Ctx
			cosetGenOne, 	// CosetGen
			1,     				// BatchSize
			KNN,     			// Ordering
			false,    		// areInputsOnDevice
			false,    		// areOutputsOnDevice
			false, 				// IsAsync
	}
}

func NttCheck[S FieldInter](input cuda_runtime.HostOrDeviceSlice[any, any], cfg *NTTConfig[S], output cuda_runtime.HostOrDeviceSlice[any, any]) {
	inputLen, outputLen := input.Len(), output.Len()
	if inputLen != outputLen {
		errorString := fmt.Sprintf(
				"input and output lengths %d; %d do not match",
				inputLen,
				outputLen,
		)
		panic(errorString)
	}
	cfg.areInputsOnDevice = input.IsOnDevice()
	cfg.areOutputsOnDevice = output.IsOnDevice()
}
