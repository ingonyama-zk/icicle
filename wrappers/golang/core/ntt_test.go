package core

import (
	"testing"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core/internal"
	"github.com/stretchr/testify/assert"
)

func TestNTTDefaultConfig(t *testing.T) {
	var cosetGenField internal.MockBaseField
	cosetGenField.One()
	var cosetGen [1]uint32
	copy(cosetGen[:], cosetGenField.GetLimbs())

	actual := GetDefaultNTTConfig(cosetGen)
	expected := NTTConfig[[1]uint32]{
		unsafe.Pointer(nil), // Ctx
		cosetGen,            // CosetGen
		1,                   // BatchSize
		false,               // ColumnsBatch
		KNN,                 // Ordering
		false,               // areInputsOnDevice
		false,               // areOutputsOnDevice
		false,               // IsAsync
		actual.Ext,          // ExtensionConfig
	}

	assert.Equal(t, expected, actual)
}

func TestNTTCheckHostScalars(t *testing.T) {
	var cosetGen internal.MockBaseField
	cosetGen.One()
	cfg := GetDefaultNTTConfig(&cosetGen)

	rawInput := make([]internal.MockBaseField, 10)
	var emptyField internal.MockBaseField
	emptyField.One()

	for i := range rawInput {
		rawInput[i] = emptyField
	}

	input := HostSliceFromElements[internal.MockBaseField](rawInput)
	output := HostSliceFromElements[internal.MockBaseField](rawInput)
	assert.NotPanics(t, func() { NttCheck(input, &cfg, output) })
	assert.False(t, cfg.areInputsOnDevice)
	assert.False(t, cfg.areOutputsOnDevice)

	rawInputLarger := make([]internal.MockBaseField, 11)
	for i := range rawInputLarger {
		rawInputLarger[i] = emptyField
	}
	output2 := HostSliceFromElements[internal.MockBaseField](rawInputLarger)
	assert.Panics(t, func() { NttCheck(input, &cfg, output2) })
}

func TestNTTCheckDeviceScalars(t *testing.T) {
	var cosetGen internal.MockBaseField
	cosetGen.One()
	cfg := GetDefaultNTTConfig(cosetGen)

	numFields := 10
	rawInput := make([]internal.MockBaseField, numFields)
	for i := range rawInput {
		var emptyField internal.MockBaseField
		emptyField.One()

		rawInput[i] = emptyField
	}

	hostElements := HostSliceFromElements[internal.MockBaseField](rawInput)

	var input DeviceSlice
	hostElements.CopyToDevice(&input, true)

	fieldBytesSize := hostElements.SizeOfElement()
	var output DeviceSlice
	output.Malloc(fieldBytesSize, numFields)

	assert.NotPanics(t, func() { NttCheck(input, &cfg, output) })
	assert.True(t, cfg.areInputsOnDevice)
	assert.True(t, cfg.areOutputsOnDevice)

	var output2 DeviceSlice
	output2.Malloc(fieldBytesSize, numFields+1)
	assert.Panics(t, func() { NttCheck(input, &cfg, output2) })
}

// TODO add check for batches and batchSize
