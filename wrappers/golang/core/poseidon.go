package core

import (
	"fmt"
	"unsafe"

	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

type PoseidonConfig struct {
	/// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
	Ctx                cr.DeviceContext
	areInputsOnDevice  bool
	areOutputsOnDevice bool
	///If true, input is considered to be a states vector, holding the preimages in aligned or not aligned format.
	///Memory under the input pointer will be used for states. If false, fresh states memory will be allocated and input will be copied into it */
	InputIsAState bool
	/// If true - input should be already aligned for poseidon permutation.
	///* Aligned format: [0, A, B, 0, C, D, ...] (as you might get by using loop_state)
	///* not aligned format: [A, B, 0, C, D, 0, ...] (as you might get from cudaMemcpy2D) */
	Aligned bool
	///If true, hash results will also be copied in the input pointer in aligned format
	LoopState bool
	///Whether to run the Poseidon asynchronously. If set to `true`, the poseidon_hash function will be
	///non-blocking and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
	///If set to false, the poseidon_hash function will block the current CPU thread. */
	IsAsync bool
}

type PoseidonConstants[T any] struct {
	Arity           int32
	PartialRounds   int32
	FullRoundsHalf  int32
	RoundConstants  unsafe.Pointer
	MdsMatrix       unsafe.Pointer
	NonSparseMatrix unsafe.Pointer
	SparseMatrices  unsafe.Pointer
	DomainTag       T
}

func GetDefaultPoseidonConfig() PoseidonConfig {
	ctx, _ := cr.GetDefaultDeviceContext()
	return PoseidonConfig{
		ctx,   // Ctx
		false, // areInputsOnDevice
		false, // areOutputsOnDevice
		false, // inputIsAState
		false, // aligned
		false, // loopState
		false, // IsAsync
	}
}

func PoseidonCheck[T any](input, output HostOrDeviceSlice, cfg *PoseidonConfig, constants *PoseidonConstants[T], arity, numberOfStates int) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer) {
	inputLen, outputLen := input.Len(), output.Len()

	if arity != int(constants.Arity) {
		errorString := fmt.Sprintf(
			"arity: %d does not match Poseidon Constants: %d",
			arity,
			outputLen,
		)
		panic(errorString)
	}

	if inputLen != arity*numberOfStates {
		errorString := fmt.Sprintf(
			"input is not the right length for the given parameters: %d, should be: %d",
			inputLen,
			arity*numberOfStates,
		)
		panic(errorString)
	}

	if outputLen != numberOfStates {
		errorString := fmt.Sprintf(
			"output is not the right length for the given parameters:%d, should be: %d",
			outputLen,
			numberOfStates,
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

	cfgPointer := unsafe.Pointer(cfg)

	return input.AsUnsafePointer(), output.AsUnsafePointer(), cfgPointer
}
