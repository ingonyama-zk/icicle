package bn254

// #cgo CFLAGS: -I./include/
// #include "curve.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

type Projective struct {
	X, Y, Z BaseField
}

func (p Projective) Size() int {
	return p.X.Size() * 3
}

func (p Projective) AsPointer() *uint32 {
	return p.X.AsPointer()
}

func (p *Projective) Zero() Projective {
	p.X.Zero()
	p.Y.One()
	p.Z.Zero()

	return *p
}

func (p *Projective) FromLimbs(x, y, z []uint32) Projective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p *Projective) FromAffine(a Affine) Projective {
	z := BaseField{}
	z.One()

	p.X = a.X
	p.Y = a.Y
	p.Z = z

	return *p
}

func (p Projective) ProjectiveEq(p2 *Projective) bool {
	cP := (*C.projective_t)(unsafe.Pointer(&p))
	cP2 := (*C.projective_t)(unsafe.Pointer(&p2))
	__ret := C.bn254_eq(cP, cP2)
	return __ret == (C._Bool)(true)
}

func (p *Projective) ProjectiveToAffine() Affine {
	var a Affine

	cA := (*C.affine_t)(unsafe.Pointer(&a))
	cP := (*C.projective_t)(unsafe.Pointer(&p))
	C.bn254_to_affine(cP, cA)
	return a
}

func GenerateProjectivePoints(size int) core.HostSlice[Projective] {
	points := make([]Projective, size)
	for i := range points {
		points[i] = Projective{}
	}

	pointsSlice := core.HostSliceFromElements[Projective](points)
	pPoints := (*C.projective_t)(unsafe.Pointer(&pointsSlice[0]))
	cSize := (C.int)(size)
	C.bn254_generate_projective_points(pPoints, cSize)

	return pointsSlice
}

type Affine struct {
	X, Y BaseField
}

func (a Affine) Size() int {
	return a.X.Size() * 2
}

func (a Affine) AsPointer() *uint32 {
	return a.X.AsPointer()
}

func (a *Affine) Zero() Affine {
	a.X.Zero()
	a.Y.Zero()

	return *a
}

func (a *Affine) FromLimbs(x, y []uint32) Affine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a Affine) ToProjective() Projective {
	var z BaseField

	return Projective{
		X: a.X,
		Y: a.Y,
		Z: z.One(),
	}
}

func AffineFromProjective(p *Projective) Affine {
	return p.ProjectiveToAffine()
}

func GenerateAffinePoints(size int) core.HostSlice[Affine] {
	points := make([]Affine, size)
	for i := range points {
		points[i] = Affine{}
	}

	pointsSlice := core.HostSliceFromElements[Affine](points)
	cPoints := (*C.affine_t)(unsafe.Pointer(&pointsSlice[0]))
	cSize := (C.int)(size)
	C.bn254_generate_affine_points(cPoints, cSize)

	return pointsSlice
}

func convertAffinePointsMontgomery(points *core.DeviceSlice, isInto bool) runtime.EIcicleError {
	defaultCfg := core.DefaultVecOpsConfig()
	cValues, _, _, cCfg, cSize := core.VecOpCheck(*points, *points, *points, &defaultCfg)
	cErr := C.bn254_affine_convert_montgomery((*C.affine_t)(cValues), (C.size_t)(cSize), (C._Bool)(isInto), (*C.VecOpsConfig)(cCfg), (*C.affine_t)(cValues))
	err := runtime.EIcicleError(cErr)
	return err
}

func AffineToMontgomery(points *core.DeviceSlice) runtime.EIcicleError {
	points.CheckDevice()
	return convertAffinePointsMontgomery(points, true)
}

func AffineFromMontgomery(points *core.DeviceSlice) runtime.EIcicleError {
	points.CheckDevice()
	return convertAffinePointsMontgomery(points, false)
}

func convertProjectivePointsMontgomery(points *core.DeviceSlice, isInto bool) runtime.EIcicleError {
	defaultCfg := core.DefaultVecOpsConfig()
	cValues, _, _, cCfg, cSize := core.VecOpCheck(*points, *points, *points, &defaultCfg)
	cErr := C.bn254_projective_convert_montgomery((*C.projective_t)(cValues), (C.size_t)(cSize), (C._Bool)(isInto), (*C.VecOpsConfig)(cCfg), (*C.projective_t)(cValues))
	err := runtime.EIcicleError(cErr)
	return err
}

func ProjectiveToMontgomery(points *core.DeviceSlice) runtime.EIcicleError {
	points.CheckDevice()
	return convertProjectivePointsMontgomery(points, true)
}

func ProjectiveFromMontgomery(points *core.DeviceSlice) runtime.EIcicleError {
	points.CheckDevice()
	return convertProjectivePointsMontgomery(points, false)
}
