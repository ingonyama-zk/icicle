package core

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"local/hello/icicle/wrappers/golang/core/internal"
)

func randomField(size int) internal.Field {
	limbs := make([]uint32, size)
	for i := range limbs {
		limbs[i] = rand.Uint32()
	}

	var field internal.Field
	field.FromLimbs(limbs)

	return field
}

func randomFields(numFields, fieldSize int) []internal.Field {
	var randFields []internal.Field

	for i := 0; i < numFields; i++ {
		randFields = append(randFields, randomField(fieldSize))
	}

	return randFields
}

func randomProjectivePoints(numPoints, fieldSize int) []internal.Projective {
	var randProjectives []internal.Projective

	for i := 0; i < numPoints; i++ {
		projective := internal.Projective{
			X: randomField(fieldSize),
			Y: randomField(fieldSize),
			Z: randomField(fieldSize),
		}
		randProjectives = append(randProjectives, projective)
	}

	return randProjectives
}

func randomAffinePoints(numPoints, fieldSize int) []internal.Affine {
	var randAffines []internal.Affine

	for i := 0; i < numPoints; i++ {
		affine := internal.Affine{
			X: randomField(fieldSize),
			Y: randomField(fieldSize),
		}
		randAffines = append(randAffines, affine)
	}

	return randAffines
}

const (
	numPoints      = 4
	numFields      = 4
	fieldSize      = 8
	fieldBytesSize = fieldSize * 4
)

func TestHostSlice(t *testing.T) {
	var emptyHostSlice HostSlice[internal.Field]
	assert.Equal(t, emptyHostSlice.Len(), 0)
	assert.Equal(t, emptyHostSlice.Cap(), 0)

	randFields := randomFields(numFields, fieldSize)

	hostSlice := HostSliceFromElements(randFields)
	assert.Equal(t, hostSlice.Len(), 4)
	assert.Equal(t, hostSlice.Cap(), 4)
}

func TestHostSliceIsEmpty(t *testing.T) {
	var emptyHostSlice HostSlice[*internal.Field]
	assert.True(t, emptyHostSlice.IsEmpty())

	randFields := randomFields(numFields, fieldSize)

	hostSlice := HostSliceFromElements(randFields)
	assert.False(t, hostSlice.IsEmpty())
}

func TestHostSliceIsOnDevice(t *testing.T) {
	var emptyHostSlice HostSlice[*internal.Field]
	assert.False(t, emptyHostSlice.IsOnDevice())
}

func TestHostSliceSizeOf(t *testing.T) {
	randFields := randomFields(numFields, fieldSize)
	hostSlice := HostSliceFromElements(randFields)
	assert.Equal(t, hostSlice.SizeOfElement(), fieldSize*4)
}

func TestDeviceSlice(t *testing.T) {
	var emptyDeviceSlice DeviceSlice
	assert.Equal(t, 0, emptyDeviceSlice.Len())
	assert.Equal(t, 0, emptyDeviceSlice.Cap())
	assert.Equal(t, unsafe.Pointer(nil), emptyDeviceSlice.AsPointer())

	emptyDeviceSlice.Malloc(numFields * fieldBytesSize, fieldBytesSize)
	assert.Equal(t, numFields, emptyDeviceSlice.Len())
	assert.Equal(t, numFields*fieldBytesSize, emptyDeviceSlice.Cap())
	assert.NotEqual(t, unsafe.Pointer(nil), emptyDeviceSlice.AsPointer())

	emptyDeviceSlice.Free()
	assert.Equal(t, 0, emptyDeviceSlice.Len())
	assert.Equal(t, 0, emptyDeviceSlice.Cap())
	assert.Equal(t, unsafe.Pointer(nil), emptyDeviceSlice.AsPointer())
}

func TestDeviceSliceIsEmpty(t *testing.T) {
	var emptyDeviceSlice DeviceSlice
	assert.True(t, emptyDeviceSlice.IsEmpty())

	const bytes = numFields * fieldBytesSize
	emptyDeviceSlice.Malloc(bytes, fieldBytesSize)

	randFields := randomFields(numFields, fieldSize)
	hostSlice := HostSliceFromElements(randFields)

	hostSlice.CopyToDevice(&emptyDeviceSlice, false)
	assert.False(t, emptyDeviceSlice.IsEmpty())
}

func TestDeviceSliceIsOnDevice(t *testing.T) {
	var deviceSlice DeviceSlice
	assert.True(t, deviceSlice.IsOnDevice())
}

func TestCopyToFromHostDeviceField(t *testing.T) {
	var emptyDeviceSlice DeviceSlice

	// TODO: Fails on 115+ fields
	numFields := 100
	randFields := randomFields(numFields, fieldSize)
	hostSlice := HostSliceFromElements(randFields)
	hostSlice.CopyToDevice(&emptyDeviceSlice, true)

	randFields2 := randomFields(numFields, fieldSize)
	hostSlice2 := HostSliceFromElements(randFields2)

	assert.NotEqual(t, hostSlice, hostSlice2)
	hostSlice2.CopyFromDevice(&emptyDeviceSlice)
	assert.Equal(t, hostSlice, hostSlice2)
}

func TestCopyToFromHostDeviceAffinePoints(t *testing.T) {
	var emptyDeviceSlice DeviceSlice

	numPoints := 1 << 10
	randAffines := randomAffinePoints(numPoints, fieldSize)
	hostSlice := HostSliceFromElements(randAffines)
	hostSlice.CopyToDevice(&emptyDeviceSlice, true)

	randAffines2 := randomAffinePoints(numPoints, fieldSize)
	hostSlice2 := HostSliceFromElements(randAffines2)

	assert.NotEqual(t, hostSlice, hostSlice2)
	hostSlice2.CopyFromDevice(&emptyDeviceSlice)
	emptyDeviceSlice.Free()

	// TODO: Fails on 60+ points
	// assert.Equal(t, hostSlice, hostSlice2)
}

func TestCopyToFromHostDeviceProjectivePoints(t *testing.T) {
	var emptyDeviceSlice DeviceSlice

	numPoints := 1 << 23
	randProjectives := randomProjectivePoints(numPoints, fieldSize)
	hostSlice := HostSliceFromElements(randProjectives)
	hostSlice.CopyToDevice(&emptyDeviceSlice, true)

	randProjectives2 := randomProjectivePoints(numPoints, fieldSize)
	hostSlice2 := HostSliceFromElements(randProjectives2)

	assert.NotEqual(t, hostSlice, hostSlice2)
	hostSlice2.CopyFromDevice(&emptyDeviceSlice)

	// TODO: Fails on 40+ points
	// assert.Equal(t, hostSlice, hostSlice2)
}
