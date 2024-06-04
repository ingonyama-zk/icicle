package core

import (
	"fmt"

	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

type SpongeConfig struct {
	/// Details related to the device such as its id and stream.
	Ctx cr.DeviceContext

	areInputsOnDevice  bool
	areResultsOnDevice bool

	InputRate  uint32
	OutputRate uint32
	Offset     uint32

	/// If true - input should be already aligned for poseidon permutation.
	/// Aligned format: [0, A, B, 0, C, D, ...] (as you might get by using loop_state)
	/// not aligned format: [A, B, 0, C, D, 0, ...] (as you might get from cudaMemcpy2D)
	RecursiveSqueeze bool

	/// If true, hash results will also be copied in the input pointer in aligned format
	Aligned bool

	/// Whether to run the MSM asynchronously. If set to `true`, the MSM function will be non-blocking
	/// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
	/// If set to `false`, the MSM function will block the current CPU thread.
	IsAsync bool
}

func GetDefaultSpongeConfig() SpongeConfig {
	ctx, _ := cr.GetDefaultDeviceContext()
	return SpongeConfig{
		ctx,
		false,
		false,
		0,
		0,
		0,
		false,
		false,
		false,
	}
}

func SpongeInputCheck(inputs HostOrDeviceSlice, numberOfStates, inputBlockLength, inputRate uint32, ctx *cr.DeviceContext) {
	if inputBlockLength > inputRate {
		errorString := fmt.Sprintf(
			"Input block (%d) can't be greater than input rate (%d)",
			inputBlockLength,
			inputRate,
		)
		panic(errorString)
	}
	inputsSizeExpected := inputBlockLength * numberOfStates
	if inputs.Len() < int(inputsSizeExpected) {
		errorString := fmt.Sprintf(
			"inputs len is %d; but needs to be at least %d",
			inputs.Len(),
			inputsSizeExpected,
		)
		panic(errorString)
	}
	if inputs.IsOnDevice() {
		inputs.(DeviceSlice).CheckDevice()
	}
}

func SpongeStatesCheck(states DeviceSlice, numberOfStates, width uint32, ctx *cr.DeviceContext) {

	statesSizeExpected := width * numberOfStates
	if states.Len() < int(statesSizeExpected) {
		errorString := fmt.Sprintf(
			"inputs len is %d; but needs to be at least %d",
			states.Len(),
			statesSizeExpected,
		)
		panic(errorString)
	}
	states.CheckDevice()
}

func SpongeOutputsCheck(outputs HostOrDeviceSlice, numberOfStates, outputLen, width uint32, recursive bool, ctx *cr.DeviceContext) {
	var outputsSizeExpected uint32
	if recursive {
		outputsSizeExpected = width * numberOfStates
	} else {
		outputsSizeExpected = outputLen * numberOfStates
	}

	if outputs.Len() < int(outputsSizeExpected) {
		errorString := fmt.Sprintf(
			"outputs len is %d; but needs to be at least %d",
			outputs.Len(),
			outputsSizeExpected,
		)
		panic(errorString)
	}
	if outputs.IsOnDevice() {
		outputs.(DeviceSlice).CheckDevice()
	}
}
