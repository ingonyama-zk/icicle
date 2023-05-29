package bn254

import (
	"encoding/binary"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/assert"

	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
)

func TestNewFieldBN254One(t *testing.T) {
	oneField := NewFieldBN254One()
	rawOneField := [8]uint32([8]uint32{0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0})

	assert.Equal(t, oneField.limbs(), rawOneField)
}

func TestNewFieldBN254Zero(t *testing.T) {
	zeroField := NewFieldBN254Zero()
	rawZeroField := [8]uint32([8]uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0})

	assert.Equal(t, zeroField.limbs(), rawZeroField)
}

func TestFieldBN254FromGnark(t *testing.T) {
	var rand fr.Element
	rand.SetRandom()

	s := FieldBN254FromGnark(rand)

	assert.Equal(t, s.limbs(), ConvertUint64ArrToUint32Arr(rand))
}

func TestFieldBN254GetLimbs(t *testing.T) {
	zeroField := NewFieldBN254Zero()

	var field FieldBN254

	assert.Equal(t, zeroField.limbs(), field.limbs())
}

func TestFieldBN254ToBytesLe(t *testing.T) {
	var rand fr.Element
	rand.SetRandom()

	f := FieldBN254FromGnark(rand)

	expected := make([]byte, len(f.s)*4) // each uint32 takes 4 bytes
	for i, v := range f.s {
		binary.LittleEndian.PutUint32(expected[i*4:], v)
	}

	assert.Equal(t, f.toBytesLe(), expected)
	assert.Equal(t, len(f.toBytesLe()), 32)
}

func TestNewPointBN254Zero(t *testing.T) {
	point := NewPointBN254Zero()

	assert.Equal(t, point.x, *NewFieldBN254Zero())
	assert.Equal(t, point.y, *NewFieldBN254One())
	assert.Equal(t, point.z, *NewFieldBN254Zero())
}

func TestBN254Eq(t *testing.T) {
	p1 := NewPointBN254Zero()
	p2 := NewPointBN254Zero()
	p3 := &PointBN254{
		x: *NewFieldBN254One(),
		y: *NewFieldBN254One(),
		z: *NewFieldBN254One(),
	}

	assert.Equal(t, p1.eq(p2), true)
	assert.Equal(t, p1.eq(p3), false)
}

func TestBN254StripZ(t *testing.T) {
	p1 := NewPointBN254Zero()
	p2ZLess := p1.strip_z()

	assert.IsType(t, PointAffineNoInfinityBN254{}, *p2ZLess)
	assert.Equal(t, p1.x, p2ZLess.x)
	assert.Equal(t, p1.y, p2ZLess.y)
}

func TestPointBN254FromGnark(t *testing.T) {
	gnarkP, _ := randG1Jac()

	p := PointBN254FromGnark(&gnarkP)

	z_inv := new(fp.Element)
	z_invsq := new(fp.Element)
	z_invq3 := new(fp.Element)
	x := new(fp.Element)
	y := new(fp.Element)

	z_inv.Inverse(&gnarkP.Z)
	z_invsq.Mul(z_inv, z_inv)
	z_invq3.Mul(z_invsq, z_inv)

	x.Mul(&gnarkP.X, z_invsq)
	y.Mul(&gnarkP.Y, z_invq3)

	assert.Equal(t, p.x, *FieldBN254FromGnark(*x))
	assert.Equal(t, p.y, *FieldBN254FromGnark(*y))
	assert.Equal(t, p.z, *NewFieldBN254One())
}

func TestPointBN254fromLimbs(t *testing.T) {
	gnarkP, _ := randG1Jac()
	p := PointBN254FromGnark(&gnarkP)

	x := p.x.limbs()
	y := p.y.limbs()
	z := p.z.limbs()

	xSlice := x[:]
	ySlice := y[:]
	zSlice := z[:]

	pFromLimbs := PointBN254fromLimbs(&xSlice, &ySlice, &zSlice)

	assert.Equal(t, pFromLimbs, p)
}

func TestNewPointAffineNoInfinityBN254Zero(t *testing.T) {
	zeroP := NewPointAffineNoInfinityBN254Zero()

	assert.Equal(t, zeroP.x, *NewFieldBN254Zero())
	assert.Equal(t, zeroP.y, *NewFieldBN254Zero())
}

func TestPointAffineNoInfinityBN254GetLimbs(t *testing.T) {
	gnarkP, _ := randG1Jac()
	p := PointBN254FromGnark(&gnarkP).strip_z()
	sliceX := p.x.limbs()
	sliceY := p.y.limbs()

	assert.Equal(t, p.limbs(), append(sliceX[:], sliceY[:]...))
}

func TestPointAffineNoInfinityBN254ToProjective(t *testing.T) {
	gnarkP, _ := randG1Jac()
	affine := PointBN254FromGnark(&gnarkP).strip_z()
	proj := affine.toProjective()

	assert.Equal(t, proj.x, affine.x)
	assert.Equal(t, proj.x, affine.x)
	assert.Equal(t, proj.z, *NewFieldBN254One())
}

func TestPointAffineNoInfinityBN254FromLimbs(t *testing.T) {
	// Initialize your test values
	x := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	y := []uint32{9, 10, 11, 12, 13, 14, 15, 16}

	// Execute your function
	result := PointAffineNoInfinityBN254FromLimbs(&x, &y)

	// Define your expected result
	expected := &PointAffineNoInfinityBN254{
		x: FieldBN254{s: getFixedLimbs(&x)},
		y: FieldBN254{s: getFixedLimbs(&y)},
	}

	// Test if result is as expected
	assert.Equal(t, result, expected)
}

func TestToGnarkAffine(t *testing.T) {
	gJac, _ := randG1Jac()
	proj := PointBN254FromGnark(&gJac)

	var gAffine bn254.G1Affine
	gAffine.FromJacobian(&gJac)

	affine := *proj.toGnarkAffine()

	assert.Equal(t, affine, gAffine)
}

func TestGetFixedLimbs(t *testing.T) {
	t.Run("case of valid input of length less than 8", func(t *testing.T) {
		slice := []uint32{1, 2, 3, 4, 5, 6, 7}
		expected := [8]uint32{1, 2, 3, 4, 5, 6, 7, 0}

		result := getFixedLimbs(&slice)
		assert.Equal(t, result, expected)
	})

	t.Run("case of valid input of length 8", func(t *testing.T) {
		slice := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
		expected := [8]uint32{1, 2, 3, 4, 5, 6, 7, 8}

		result := getFixedLimbs(&slice)
		assert.Equal(t, result, expected)
	})

	t.Run("case of empty input", func(t *testing.T) {
		slice := []uint32{}
		expected := [8]uint32{0, 0, 0, 0, 0, 0, 0, 0}

		result := getFixedLimbs(&slice)
		assert.Equal(t, result, expected)
	})

	t.Run("case of input length greater than 8", func(t *testing.T) {
		slice := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		defer func() {
			if r := recover(); r == nil {
				t.Errorf("the code did not panic")
			}
		}()

		getFixedLimbs(&slice)
	})
}
