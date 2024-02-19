package bn254

// import (
// 	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
// 	"testing"

// 	"github.com/stretchr/testify/assert"
// )

// func TestNewProjective(t *testing.T) {
// 	expected := Projective{
// 		X: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 		Y: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 		Z: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 	}

// 	var actual Projective

// 	assert.Equal(t, expected, actual)
// }

// func TestProjectiveEq(t *testing.T) {
// 	fieldOne := BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}}
// 	fieldOne.One()
// 	expected := Projective{
// 		X: fieldOne,
// 		Y: fieldOne,
// 		Z: fieldOne,
// 	}

// 	var actual Projective
// 	actual.X.One()
// 	actual.Y.One()
// 	actual.Z.One()

// 	assert.True(t, actual.ProjectiveEq(&expected))
// }

// func TestProjectiveToAffine(t *testing.T) {
// 	t.FailNow() // hanging on call to C++
// 	expected := Affine{
// 		X: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 		Y: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 	}

// 	var projective Projective
// 	actual := projective.ProjectiveToAffine()

// 	assert.Equal(t, expected, actual)
// }

// func TestGenerateProjectivePoints(t *testing.T) {
// 	const numPoints = 1 << 20
// 	points := GenerateProjectivePoints(numPoints)

// 	assert.Implements(t, (*core.HostOrDeviceSlice)(nil), &points)

// 	assert.Equal(t, numPoints, points.Len())
// 	var zeroPoint Projective
// 	assert.NotContains(t, points, zeroPoint)
// }

// func TestProjectiveMongtomeryConversion(t *testing.T) {
// 	size := 1 << 20
// 	points := GenerateProjectivePoints(size)

// 	var devicePoints core.DeviceSlice
// 	points.CopyToDevice(&devicePoints, true)

// 	ProjectiveToMontgomery(&devicePoints)

// 	pointsMontHost := GenerateProjectivePoints(size)

// 	pointsMontHost.CopyFromDevice(&devicePoints)
// 	assert.NotEqual(t, points, pointsMontHost)

// 	ProjectiveFromMontgomery(&devicePoints)

// 	pointsMontHost.CopyFromDevice(&devicePoints)
// 	assert.Equal(t, points, pointsMontHost)
// }

// func TestNewAffine(t *testing.T) {
// 	expected := Affine{
// 		X: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 		Y: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 	}

// 	var actual Affine

// 	assert.Equal(t, expected, actual)
// }

// func TestAffineFromProjective(t *testing.T) {
// 	t.FailNow() // hangs on call to C++
// 	expected := Affine{
// 		X: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 		Y: BaseField{limbs: [BASE_LIMBS]uint32{0, 0, 0, 0, 0, 0, 0, 0}},
// 	}

// 	var projective Projective
// 	actual := AffineFromProjective(&projective)

// 	assert.Equal(t, expected, actual)
// }

// func TestGenerateAffinePoints(t *testing.T) {
// 	numPoints := 1 << 20
// 	points := GenerateAffinePoints(numPoints)

// 	assert.Implements(t, (*core.HostOrDeviceSlice)(nil), &points)

// 	assert.Equal(t, numPoints, points.Len())
// 	var zeroPoint Projective
// 	assert.NotContains(t, points, zeroPoint)
// }

// func TestAffineMongtomeryConversion(t *testing.T) {
// 	size := 1 << 20
// 	points := GenerateAffinePoints(size)

// 	var devicePoints core.DeviceSlice
// 	points.CopyToDevice(&devicePoints, true)

// 	AffineToMontgomery(&devicePoints)

// 	pointsMontHost := GenerateAffinePoints(size)

// 	pointsMontHost.CopyFromDevice(&devicePoints)
// 	assert.NotEqual(t, points, pointsMontHost)

// 	AffineFromMontgomery(&devicePoints)

// 	pointsMontHost.CopyFromDevice(&devicePoints)
// 	assert.Equal(t, points, pointsMontHost)
// }
