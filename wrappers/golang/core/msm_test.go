package core

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core/internal"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"

	"github.com/stretchr/testify/assert"
)

func TestMSMDefaultConfig(t *testing.T) {
	ctx, _ := cr.GetDefaultDeviceContext()
	expected := MSMConfig{
		ctx,   // Ctx
		0,     // pointsSize
		1,     // PrecomputeFactor
		0,     // C
		0,     // Bitsize
		10,    // LargeBucketFactor
		1,     // batchSize
		false, // areScalarsOnDevice
		false, // AreScalarsMontgomeryForm
		false, // arePointsOnDevice
		false, // ArePointsMontgomeryForm
		false, // areResultsOnDevice
		false, // IsBigTriangle
		false, // IsAsync
	}

	actual := GetDefaultMSMConfig()

	assert.Equal(t, expected, actual)
}

func TestMSMCheckHostSlices(t *testing.T) {
	cfg := GetDefaultMSMConfig()

	rawScalars := make([]internal.MockField, 10)
	for i := range rawScalars {
		var emptyField internal.MockField
		emptyField.One()

		rawScalars[i] = emptyField
	}
	scalars := HostSliceFromElements[internal.MockField](rawScalars)

	affine := internal.MockAffine{}
	var emptyField internal.MockField
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
	assert.False(t, cfg.arePointsOnDevice)
	assert.False(t, cfg.areResultsOnDevice)
	assert.Equal(t, int32(1), cfg.batchSize)

	output2 := make(HostSlice[internal.MockProjective], 3)
	assert.Panics(t, func() { MsmCheck(scalars, points, &cfg, output2) })
}

func TestMSMCheckDeviceSlices(t *testing.T) {
	cfg := GetDefaultMSMConfig()

	rawScalars := make([]internal.MockField, 10)
	for i := range rawScalars {
		var emptyField internal.MockField
		emptyField.One()

		rawScalars[i] = emptyField
	}
	scalars := HostSliceFromElements[internal.MockField](rawScalars)
	var scalarsOnDevice DeviceSlice
	scalars.CopyToDevice(&scalarsOnDevice, true)

	affine := internal.MockAffine{}
	var emptyField internal.MockField
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
	assert.True(t, cfg.arePointsOnDevice)
	assert.False(t, cfg.areResultsOnDevice)
	assert.Equal(t, int32(1), cfg.batchSize)

	output2 := make(HostSlice[internal.MockProjective], 3)
	assert.Panics(t, func() { MsmCheck(scalarsOnDevice, pointsOnDevice, &cfg, output2) })
}

// TODO add check for batches and batchSize
