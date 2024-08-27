package core

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core/internal"
	"github.com/stretchr/testify/assert"
)

func TestMSMDefaultConfig(t *testing.T) {
	actual := GetDefaultMSMConfig()

	expected := MSMConfig{
		StreamHandle:             nil,
		PrecomputeFactor:         1,
		C:                        0,
		Bitsize:                  0,
		BatchSize:                1,
		AreBasesShared:           true,
		areScalarsOnDevice:       false,
		AreScalarsMontgomeryForm: false,
		areBasesOnDevice:         false,
		AreBasesMontgomeryForm:   false,
		areResultsOnDevice:       false,
		IsAsync:                  false,
		Ext:                      actual.Ext,
	}

	assert.EqualValues(t, expected, actual)
}

func TestMSMCheckHostSlices(t *testing.T) {
	cfg := GetDefaultMSMConfig()

	rawScalars := make([]internal.MockBaseField, 10)
	for i := range rawScalars {
		var emptyField internal.MockBaseField
		emptyField.One()

		rawScalars[i] = emptyField
	}
	scalars := HostSliceFromElements[internal.MockBaseField](rawScalars)

	affine := internal.MockAffine{}
	var emptyField internal.MockBaseField
	emptyField.One()
	affine.FromLimbs(emptyField.GetLimbs(), emptyField.GetLimbs())
	rawAffinePoints := make([]internal.MockAffine, 10)
	for i := range rawAffinePoints {
		rawAffinePoints[i] = affine
	}
	points := HostSliceFromElements[internal.MockAffine](rawAffinePoints)

	output := make(HostSlice[internal.MockProjective], 1)
	assert.NotPanics(t, func() { MsmCheck(scalars, points, &cfg, output) })
	assert.False(t, cfg.areScalarsOnDevice)
	assert.False(t, cfg.areBasesOnDevice)
	assert.False(t, cfg.areResultsOnDevice)
	assert.Equal(t, int32(1), cfg.BatchSize)

	output2 := make(HostSlice[internal.MockProjective], 3)
	assert.Panics(t, func() { MsmCheck(scalars, points, &cfg, output2) })
}

func TestMSMCheckDeviceSlices(t *testing.T) {
	cfg := GetDefaultMSMConfig()

	rawScalars := make([]internal.MockBaseField, 10)
	for i := range rawScalars {
		var emptyField internal.MockBaseField
		emptyField.One()

		rawScalars[i] = emptyField
	}
	scalars := HostSliceFromElements[internal.MockBaseField](rawScalars)
	var scalarsOnDevice DeviceSlice
	scalars.CopyToDevice(&scalarsOnDevice, true)

	affine := internal.MockAffine{}
	var emptyField internal.MockBaseField
	emptyField.One()
	affine.FromLimbs(emptyField.GetLimbs(), emptyField.GetLimbs())
	rawAffinePoints := make([]internal.MockAffine, 10)
	for i := range rawAffinePoints {
		rawAffinePoints[i] = affine
	}
	points := HostSliceFromElements[internal.MockAffine](rawAffinePoints)
	var pointsOnDevice DeviceSlice
	points.CopyToDevice(&pointsOnDevice, true)

	output := make(HostSlice[internal.MockProjective], 1)
	assert.NotPanics(t, func() { MsmCheck(scalarsOnDevice, pointsOnDevice, &cfg, output) })
	assert.True(t, cfg.areScalarsOnDevice)
	assert.True(t, cfg.areBasesOnDevice)
	assert.False(t, cfg.areResultsOnDevice)
	assert.Equal(t, int32(1), cfg.BatchSize)

	output2 := make(HostSlice[internal.MockProjective], 3)
	assert.Panics(t, func() { MsmCheck(scalarsOnDevice, pointsOnDevice, &cfg, output2) })
}
