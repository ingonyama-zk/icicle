package tests

import (
	"testing"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254"
	poseidon "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254/poseidon"
)

func TestPoseidon(t *testing.T) {

	arity := 2
	numberOfStates := 1

	cfg := poseidon.GetDefaultPoseidonConfig()

	var constants core.PoseidonConstants[bn254.ScalarField]
	ctx, _ := cr.GetDefaultDeviceContext()

	poseidon.InitOptimizedPoseidonConstantsCuda(arity, ctx, &constants) //generate constants

	scalars := bn254.GenerateScalars(numberOfStates * arity)
	scalars[0] = scalars[0].Zero()
	scalars[1] = scalars[0].Zero()

	scalarsCopy := core.HostSliceFromElements(scalars[:numberOfStates*arity])

	stream, _ := cr.CreateStream()

	var deviceInput core.DeviceSlice
	scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
	var deviceOutput core.DeviceSlice
	deviceOutput.MallocAsync(numberOfStates*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

	poseidon.PoseidonHash(deviceInput, deviceOutput, numberOfStates, arity, &cfg, &constants) //run Hash function

	output := make(core.HostSlice[bn254.ScalarField], numberOfStates)
	output.CopyFromDeviceAsync(&deviceOutput, stream)

}
