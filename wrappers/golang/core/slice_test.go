package core

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core/internal"
	"github.com/stretchr/testify/assert"
)

func randomField(size int) internal.MockField {
	limbs := make([]uint64, size)
	for i := range limbs {
		limbs[i] = rand.Uint64()
	}

	var field internal.MockField
	field.FromLimbs(limbs)

	return field
}

func randomFields(numFields, fieldSize int) []internal.MockField {
	var randFields []internal.MockField

	for i := 0; i < numFields; i++ {
		randFields = append(randFields, randomField(fieldSize))
	}

	return randFields
}

// This function is solely for the purpose of testing HostDeviceSlice
// It can produce invalid points and should not be used to test curve operations
func randomProjectivePoints(numPoints, fieldSize int) []internal.MockProjective {
	var randProjectives []internal.MockProjective

	for i := 0; i < numPoints; i++ {
		projective := internal.MockProjective{
			X: randomField(fieldSize),
			Y: randomField(fieldSize),
			Z: randomField(fieldSize),
		}
		randProjectives = append(randProjectives, projective)
	}

	return randProjectives
}

// This function is solely for the purpose of testing HostDeviceSlice
// It can produce invalid points and should not be used to test curve operations
func randomAffinePoints(numPoints, fieldSize int) []internal.MockAffine {
	var randAffines []internal.MockAffine

	for i := 0; i < numPoints; i++ {
		affine := internal.MockAffine{
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
	fieldSize      = 4
	fieldBytesSize = fieldSize * 8
)

func TestHostSlice(t *testing.T) {
	var emptyHostSlice HostSlice[internal.MockField]
	assert.Equal(t, emptyHostSlice.Len(), 0)
	assert.Equal(t, emptyHostSlice.Cap(), 0)

	randFields := randomFields(numFields, fieldSize)

	hostSlice := HostSliceFromElements(randFields)
	assert.Equal(t, hostSlice.Len(), 4)
	assert.Equal(t, hostSlice.Cap(), 4)

	hostSliceCasted := (HostSlice[internal.MockField])(randFields)
	assert.Equal(t, hostSliceCasted.Len(), 4)
	assert.Equal(t, hostSliceCasted.Cap(), 4)
}

func TestHostSliceIsEmpty(t *testing.T) {
	var emptyHostSlice HostSlice[*internal.MockField]
	assert.True(t, emptyHostSlice.IsEmpty())

	randFields := randomFields(numFields, fieldSize)

	hostSlice := HostSliceFromElements(randFields)
	assert.False(t, hostSlice.IsEmpty())
}

func TestHostSliceIsOnDevice(t *testing.T) {
	var emptyHostSlice HostSlice[*internal.MockField]
	assert.False(t, emptyHostSlice.IsOnDevice())
}

func TestHostSliceSizeOf(t *testing.T) {
	randFields := randomFields(numFields, fieldSize)
	hostSlice := HostSliceFromElements(randFields)
	assert.Equal(t, hostSlice.SizeOfElement(), fieldSize*8)
}

func TestDeviceSlice(t *testing.T) {
	var emptyDeviceSlice DeviceSlice
	assert.Equal(t, 0, emptyDeviceSlice.Len())
	assert.Equal(t, 0, emptyDeviceSlice.Cap())
	assert.Equal(t, unsafe.Pointer(nil), emptyDeviceSlice.AsUnsafePointer())

	emptyDeviceSlice.Malloc(numFields*fieldBytesSize, fieldBytesSize)
	assert.Equal(t, numFields, emptyDeviceSlice.Len())
	assert.Equal(t, numFields*fieldBytesSize, emptyDeviceSlice.Cap())
	assert.NotEqual(t, unsafe.Pointer(nil), emptyDeviceSlice.AsUnsafePointer())

	emptyDeviceSlice.Free()
	assert.Equal(t, 0, emptyDeviceSlice.Len())
	assert.Equal(t, 0, emptyDeviceSlice.Cap())
	assert.Equal(t, unsafe.Pointer(nil), emptyDeviceSlice.AsUnsafePointer())
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

	numFields := 1 << 10
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

	assert.Equal(t, hostSlice, hostSlice2)
}

func TestCopyToFromHostDeviceProjectivePoints(t *testing.T) {
	var emptyDeviceSlice DeviceSlice

	numPoints := 1 << 15
	randProjectives := randomProjectivePoints(numPoints, fieldSize)
	hostSlice := HostSliceFromElements(randProjectives)
	hostSlice.CopyToDevice(&emptyDeviceSlice, true)

	randProjectives2 := randomProjectivePoints(numPoints, fieldSize)
	hostSlice2 := HostSliceFromElements(randProjectives2)

	assert.NotEqual(t, hostSlice, hostSlice2)
	hostSlice2.CopyFromDevice(&emptyDeviceSlice)

	assert.Equal(t, hostSlice, hostSlice2)
}

func TestSliceRanges(t *testing.T) {
	var deviceSlice DeviceSlice

	numPoints := 1 << 3
	randProjectives := randomProjectivePoints(numPoints, fieldSize)
	hostSlice := (HostSlice[internal.MockProjective])(randProjectives)
	hostSlice.CopyToDevice(&deviceSlice, true)

	// RangeFrom
	var zeroProj internal.MockProjective
	hostSliceRet := HostSliceWithValue[internal.MockProjective](zeroProj, numPoints-2)

	deviceSliceRangeFrom := deviceSlice.RangeFrom(2)
	hostSliceRet.CopyFromDevice(&deviceSliceRangeFrom)
	assert.Equal(t, hostSlice[2:], hostSliceRet)

	// RangeTo
	deviceSliceRangeTo := deviceSlice.RangeTo(numPoints-3, true)
	hostSliceRet.CopyFromDevice(&deviceSliceRangeTo)
	assert.Equal(t, hostSlice[:6], hostSliceRet)

	// Range
	hostSliceRange := HostSliceWithValue[internal.MockProjective](zeroProj, numPoints-4)
	deviceSliceRange := deviceSlice.Range(2, numPoints-3, true)
	hostSliceRange.CopyFromDevice(&deviceSliceRange)
	assert.Equal(t, hostSlice[2:6], hostSliceRange)
}
