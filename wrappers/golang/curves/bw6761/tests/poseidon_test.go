package tests

import (
	"testing"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bw6_761 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761"
	poseidon "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761/poseidon"
	"github.com/stretchr/testify/assert"
)

func TestPoseidon(t *testing.T) {

	arity := 2
	numberOfStates := 1

	ctx, _ := cr.GetDefaultDeviceContext()
	p, err := poseidon.Load(uint32(arity), &ctx)
	assert.Equal(t, core.IcicleSuccess, err.IcicleErrorCode)

	cfg := p.GetDefaultSpongeConfig()

	scalars := bw6_761.GenerateScalars(numberOfStates * arity)
	scalars[0] = scalars[0].Zero()
	scalars[1] = scalars[0].Zero()

	scalarsCopy := core.HostSliceFromElements(scalars[:numberOfStates*arity])

	var deviceInput core.DeviceSlice
	scalarsCopy.CopyToDevice(&deviceInput, true)
	var deviceOutput core.DeviceSlice
	deviceOutput.Malloc(numberOfStates*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement())

	err = p.HashMany(deviceInput, deviceOutput, uint32(numberOfStates), 1, 1, &cfg) //run Hash function
	assert.Equal(t, core.IcicleSuccess, err.IcicleErrorCode)

	output := make(core.HostSlice[bw6_761.ScalarField], numberOfStates)
	output.CopyFromDevice(&deviceOutput)
}
