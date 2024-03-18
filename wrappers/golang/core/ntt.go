package core

import (
	"fmt"

	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

type NTTDir int8

const (
	KForward NTTDir = iota
	KInverse NTTDir = 1
)

type Ordering uint32

const (
	KNN Ordering = iota
	KNR Ordering = 1
	KRN Ordering = 2
	KRR Ordering = 3
	KNM Ordering = 4
	KMN Ordering = 5
)

type NTTConfig[T any] struct {
	/// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
	Ctx cr.DeviceContext
	/// Coset generator. Used to perform coset (i)NTTs. Default value: `S::one()` (corresponding to no coset being used).
	CosetGen T
	/// The number of NTTs to compute. Default value: 1.
	BatchSize int32
	/// If true the function will compute the NTTs over the columns of the input matrix and not over the rows.
	ColumnsBatch bool
	/// Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`.
	Ordering           Ordering
	areInputsOnDevice  bool
	areOutputsOnDevice bool
	/// Whether to run the NTT asynchronously. If set to `true`, the NTT function will be non-blocking and you'd need to synchronize
	/// it explicitly by running `stream.synchronize()`. If set to false, the NTT function will block the current CPU thread.
	IsAsync bool
}

func GetDefaultNTTConfig[T any](cosetGen T) NTTConfig[T] {
	ctx, _ := cr.GetDefaultDeviceContext()
	return NTTConfig[T]{
		ctx,      // Ctx
		cosetGen, // CosetGen
		1,        // BatchSize
		false,    // ColumnsBatch
		KNN,      // Ordering
		false,    // areInputsOnDevice
		false,    // areOutputsOnDevice
		false,    // IsAsync
	}
}

func NttCheck[T any](input HostOrDeviceSlice, cfg *NTTConfig[T], output HostOrDeviceSlice) {
	inputLen, outputLen := input.Len(), output.Len()
	if inputLen != outputLen {
		errorString := fmt.Sprintf(
			"input and output capacities %d; %d are not equal",
			inputLen,
			outputLen,
		)
		panic(errorString)
	}
	cfg.areInputsOnDevice = input.IsOnDevice()
	cfg.areOutputsOnDevice = output.IsOnDevice()
}
