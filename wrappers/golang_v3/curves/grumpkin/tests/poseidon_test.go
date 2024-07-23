package tests

import (
	"testing"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/cuda_runtime"
	grumpkin "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/grumpkin"
	poseidon "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/grumpkin/poseidon"
)

func TestPoseidon(t *testing.T) {

	arity := 2
	numberOfStates := 1

	cfg := poseidon.GetDefaultPoseidonConfig()
	cfg.IsAsync = true
	stream, _ := cr.CreateStream()
	cfg.Ctx.Stream = &stream

	var constants core.PoseidonConstants[grumpkin.ScalarField]

	poseidon.InitOptimizedPoseidonConstantsCuda(arity, cfg.Ctx, &constants) //generate constants

	scalars := grumpkin.GenerateScalars(numberOfStates * arity)
	scalars[0] = scalars[0].Zero()
	scalars[1] = scalars[0].Zero()

	scalarsCopy := core.HostSliceFromElements(scalars[:numberOfStates*arity])

	var deviceInput core.DeviceSlice
	scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
	var deviceOutput core.DeviceSlice
	deviceOutput.MallocAsync(numberOfStates*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

	poseidon.PoseidonHash(deviceInput, deviceOutput, numberOfStates, &cfg, &constants) //run Hash function

	output := make(core.HostSlice[grumpkin.ScalarField], numberOfStates)
	output.CopyFromDeviceAsync(&deviceOutput, stream)

}
