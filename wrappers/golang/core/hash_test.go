package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHashDefaultConfig(t *testing.T) {
	actual := GetDefaultHashConfig()

	expected := HashConfig{
		StreamHandle:             nil,
		batchSize:                1,
		areInputsOnDevice:       	false,
		areOutputsOnDevice: 			false,
		IsAsync:                  false,
		Ext:                      actual.Ext,
	}

	assert.EqualValues(t, expected, actual)
}

func TestHashCheckHostSlices(t *testing.T) {
	cfg := GetDefaultHashConfig()
	input := make([]byte, 512)
	inputHost := HostSliceFromElements(input)
	
	output := make([]byte, 32)
	outputHost := HostSliceFromElements(output)

	assert.NotPanics(t, func() { HashCheck(inputHost, outputHost, 32, &cfg) })
	assert.False(t, cfg.areInputsOnDevice)
	assert.False(t, cfg.areOutputsOnDevice)
	assert.Equal(t, uint64(1), cfg.batchSize)
}

func TestHashCheckDeviceSlices(t *testing.T) {
	cfg := GetDefaultHashConfig()
	input := make([]byte, 512)
	inputHost := HostSliceFromElements(input)
	var d_input DeviceSlice
	inputHost.CopyToDevice(&d_input, true)
	
	output := make([]byte, 32)
	outputHost := HostSliceFromElements(output)
	var d_output DeviceSlice

	outputHost.CopyToDevice(&d_output, true)

	assert.NotPanics(t, func() { HashCheck(d_input, d_output, 32, &cfg) })
	assert.True(t, cfg.areInputsOnDevice)
	assert.True(t, cfg.areOutputsOnDevice)
	assert.Equal(t, uint64(1), cfg.batchSize)
}

func TestHashCheckMixSlices(t *testing.T) {
	cfg := GetDefaultHashConfig()
	input := make([]byte, 512)
	inputHost := HostSliceFromElements(input)
	var d_input DeviceSlice
	inputHost.CopyToDevice(&d_input, true)
	
	output := make([]byte, 32)
	outputHost := HostSliceFromElements(output)

	assert.NotPanics(t, func() { HashCheck(d_input, outputHost, 32, &cfg) })
	assert.True(t, cfg.areInputsOnDevice)
	assert.False(t, cfg.areOutputsOnDevice)
	assert.Equal(t, uint64(1), cfg.batchSize)
}

func TestHashCheckBatch(t *testing.T) {
	cfg := GetDefaultHashConfig()
	input := make([]byte, 512)
	inputHost := HostSliceFromElements(input)
	var d_input DeviceSlice
	inputHost.CopyToDevice(&d_input, true)
	
	output := make([]byte, 64)
	outputHost := HostSliceFromElements(output)

	assert.NotPanics(t, func() { HashCheck(d_input, outputHost, 32, &cfg) })
	assert.True(t, cfg.areInputsOnDevice)
	assert.False(t, cfg.areOutputsOnDevice)
	assert.Equal(t, uint64(2), cfg.batchSize)
}
