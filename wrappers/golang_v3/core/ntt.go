package core

import (
	"fmt"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

type NTTDir int8

const (
	KForward NTTDir = iota
	KInverse
)

type Ordering uint32

const (
	KNN Ordering = iota
	KNR
	KRN
	KRR
	KNM
	KMN
)

type NttAlgorithm uint32

const (
	Auto NttAlgorithm = iota
	Radix2
	MixedRadix
)

const CUDA_NTT_FAST_TWIDDLES_MODE = "fast_twiddles"
const CUDA_NTT_ALGORITHM = "ntt_algorithm"

type NTTConfig[T any] struct {
	/// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
	StreamHandle runtime.Stream
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
	Ext     runtime.ConfigExtensionHandler
}

func GetDefaultNTTConfig[T any](cosetGen T) NTTConfig[T] {
	return NTTConfig[T]{
		nil,      // StreamHandle
		cosetGen, // CosetGen
		1,        // BatchSize
		false,    // ColumnsBatch
		KNN,      // Ordering
		false,    // areInputsOnDevice
		false,    // areOutputsOnDevice
		false,    // IsAsync
		nil,      // Ext
	}
}

func NttCheck[T any](input HostOrDeviceSlice, cfg *NTTConfig[T], output HostOrDeviceSlice) (unsafe.Pointer, unsafe.Pointer, int, unsafe.Pointer) {
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

	if input.IsOnDevice() {
		input.(DeviceSlice).CheckDevice()
	}

	if output.IsOnDevice() {
		output.(DeviceSlice).CheckDevice()
	}

	size := input.Len() / int(cfg.BatchSize)
	cfgPointer := unsafe.Pointer(cfg)

	return input.AsUnsafePointer(), output.AsUnsafePointer(), size, cfgPointer
}

type NTTInitDomainConfig struct {
	StreamHandle runtime.Stream
	IsAsync      bool
	Ext          runtime.ConfigExtensionHandler
}

func GetDefaultNTTInitDomainConfig() NTTInitDomainConfig {
	return NTTInitDomainConfig{
		nil,   // StreamHandle
		false, // IsAsync
		nil,   // Ext
	}
}
