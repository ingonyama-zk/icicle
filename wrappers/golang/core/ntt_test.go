package core

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core/internal"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	"github.com/stretchr/testify/assert"
)

func TestNTTDefaultConfig(t *testing.T) {
	var cosetGenField internal.MockField
	cosetGenField.One()
	var cosetGen [1]uint64
	copy(cosetGen[:], cosetGenField.GetLimbs())
	ctx, _ := cr.GetDefaultDeviceContext()
	expected := NTTConfig[[1]uint64]{
		ctx,      // Ctx
		cosetGen, // CosetGen
		1,        // BatchSize
		false,    // ColumnsBatch
		KNN,      // Ordering
		false,    // areInputsOnDevice
		false,    // areOutputsOnDevice
		false,    // IsAsync
		Auto,     // NttAlgorithm
	}

	actual := GetDefaultNTTConfig(cosetGen)

	assert.Equal(t, expected, actual)
}

func TestNTTCheckHostScalars(t *testing.T) {
	var cosetGen internal.MockField
	cosetGen.One()
	cfg := GetDefaultNTTConfig(&cosetGen)

	rawInput := make([]internal.MockField, 10)
	var emptyField internal.MockField
	emptyField.One()

	for i := range rawInput {
		rawInput[i] = emptyField
	}

	input := HostSliceFromElements[internal.MockField](rawInput)
	output := HostSliceFromElements[internal.MockField](rawInput)
	assert.NotPanics(t, func() { NttCheck(input, &cfg, output) })
	assert.False(t, cfg.areInputsOnDevice)
	assert.False(t, cfg.areOutputsOnDevice)

	rawInputLarger := make([]internal.MockField, 11)
	for i := range rawInputLarger {
		rawInputLarger[i] = emptyField
	}
	output2 := HostSliceFromElements[internal.MockField](rawInputLarger)
	assert.Panics(t, func() { NttCheck(input, &cfg, output2) })
}

func TestNTTCheckDeviceScalars(t *testing.T) {
	var cosetGen internal.MockField
	cosetGen.One()
	cfg := GetDefaultNTTConfig(cosetGen)

	numFields := 10
	rawInput := make([]internal.MockField, numFields)
	for i := range rawInput {
		var emptyField internal.MockField
		emptyField.One()

		rawInput[i] = emptyField
	}

	hostElements := HostSliceFromElements[internal.MockField](rawInput)

	var input DeviceSlice
	hostElements.CopyToDevice(&input, true)

	fieldBytesSize := hostElements.SizeOfElement()
	var output DeviceSlice
	output.Malloc(numFields*fieldBytesSize, fieldBytesSize)

	assert.NotPanics(t, func() { NttCheck(input, &cfg, output) })
	assert.True(t, cfg.areInputsOnDevice)
	assert.True(t, cfg.areOutputsOnDevice)

	var output2 DeviceSlice
	output2.Malloc((numFields+1)*fieldBytesSize, fieldBytesSize)
	assert.Panics(t, func() { NttCheck(input, &cfg, output2) })
}

// TODO add check for batches and batchSize
