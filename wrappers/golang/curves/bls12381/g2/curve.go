package g2

// #cgo CFLAGS: -I./include/
// #include "curve.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

type G2Projective struct {
	X, Y, Z G2BaseField
}

func (p G2Projective) Size() int {
	return p.X.Size() * 3
}

func (p G2Projective) AsPointer() *uint32 {
	return p.X.AsPointer()
}

func (p *G2Projective) Zero() G2Projective {
	p.X.Zero()
	p.Y.One()
	p.Z.Zero()

	return *p
}

func (p *G2Projective) FromLimbs(x, y, z []uint32) G2Projective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p *G2Projective) FromAffine(a G2Affine) G2Projective {

	cA := (*C.g2_affine_t)(unsafe.Pointer(&a))
	cP := (*C.g2_projective_t)(unsafe.Pointer(p))
	C.bls12_381_g2_from_affine(cA, cP)
	return *p
}

func (p G2Projective) ProjectiveEq(p2 *G2Projective) bool {
	cP := (*C.g2_projective_t)(unsafe.Pointer(&p))
	cP2 := (*C.g2_projective_t)(unsafe.Pointer(p2))
	__ret := C.bls12_381_g2_eq(cP, cP2)
	return __ret == (C._Bool)(true)
}

func (p G2Projective) Add(p2 *G2Projective) G2Projective {
	var res G2Projective

	cP := (*C.g2_projective_t)(unsafe.Pointer(&p))
	cP2 := (*C.g2_projective_t)(unsafe.Pointer(p2))
	cRes := (*C.g2_projective_t)(unsafe.Pointer(&res))

	C.bls12_381_g2_ecadd(cP, cP2, cRes)

	return res
}

func (p G2Projective) Sub(p2 *G2Projective) G2Projective {
	var res G2Projective

	cP := (*C.g2_projective_t)(unsafe.Pointer(&p))
	cP2 := (*C.g2_projective_t)(unsafe.Pointer(p2))
	cRes := (*C.g2_projective_t)(unsafe.Pointer(&res))

	C.bls12_381_g2_ecsub(cP, cP2, cRes)

	return res
}

func (p *G2Projective) ToAffine() G2Affine {
	var a G2Affine

	cA := (*C.g2_affine_t)(unsafe.Pointer(&a))
	cP := (*C.g2_projective_t)(unsafe.Pointer(p))
	C.bls12_381_g2_to_affine(cP, cA)
	return a
}

func G2GenerateProjectivePoints(size int) core.HostSlice[G2Projective] {
	points := make([]G2Projective, size)
	for i := range points {
		points[i] = G2Projective{}
	}

	pointsSlice := core.HostSliceFromElements[G2Projective](points)
	pPoints := (*C.g2_projective_t)(unsafe.Pointer(&pointsSlice[0]))
	cSize := (C.int)(size)
	C.bls12_381_g2_generate_projective_points(pPoints, cSize)

	return pointsSlice
}

type G2Affine struct {
	X, Y G2BaseField
}

func (a G2Affine) Size() int {
	return a.X.Size() * 2
}

func (a G2Affine) AsPointer() *uint32 {
	return a.X.AsPointer()
}

func (a *G2Affine) Zero() G2Affine {
	a.X.Zero()
	a.Y.Zero()

	return *a
}

func (a *G2Affine) IsZero() bool {
	return a.X.IsZero() && a.Y.IsZero()
}

func (a *G2Affine) FromLimbs(x, y []uint32) G2Affine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a G2Affine) ToProjective() G2Projective {
	var p G2Projective

	// TODO - Figure out why this sometimes returns an empty projective point, i.e. {x:0, y:0, z:0}
	// cA := (*C.g2_affine_t)(unsafe.Pointer(&a))
	// cP := (*C.g2_projective_t)(unsafe.Pointer(&p))
	// C.bls12_381_g2_from_affine(cA, cP)

	if a.IsZero() {
		p.Zero()
	} else {
		p.X = a.X
		p.Y = a.Y
		p.Z.One()
	}

	return p
}

func G2AffineFromProjective(p *G2Projective) G2Affine {
	return p.ToAffine()
}

func G2GenerateAffinePoints(size int) core.HostSlice[G2Affine] {
	points := make([]G2Affine, size)
	for i := range points {
		points[i] = G2Affine{}
	}

	pointsSlice := core.HostSliceFromElements[G2Affine](points)
	cPoints := (*C.g2_affine_t)(unsafe.Pointer(&pointsSlice[0]))
	cSize := (C.int)(size)
	C.bls12_381_g2_generate_affine_points(cPoints, cSize)

	return pointsSlice
}

func convertG2AffinePointsMontgomery(points core.HostOrDeviceSlice, isInto bool) runtime.EIcicleError {
	defaultCfg := core.DefaultVecOpsConfig()
	cValues, _, _, cCfg, cSize := core.VecOpCheck(points, points, points, &defaultCfg)
	cErr := C.bls12_381_g2_affine_convert_montgomery((*C.g2_affine_t)(cValues), (C.size_t)(cSize), (C._Bool)(isInto), (*C.VecOpsConfig)(cCfg), (*C.g2_affine_t)(cValues))
	err := runtime.EIcicleError(cErr)
	return err
}

func G2AffineToMontgomery(points core.HostOrDeviceSlice) runtime.EIcicleError {
	return convertG2AffinePointsMontgomery(points, true)
}

func G2AffineFromMontgomery(points core.HostOrDeviceSlice) runtime.EIcicleError {
	return convertG2AffinePointsMontgomery(points, false)
}

func convertG2ProjectivePointsMontgomery(points core.HostOrDeviceSlice, isInto bool) runtime.EIcicleError {
	defaultCfg := core.DefaultVecOpsConfig()
	cValues, _, _, cCfg, cSize := core.VecOpCheck(points, points, points, &defaultCfg)
	cErr := C.bls12_381_g2_projective_convert_montgomery((*C.g2_projective_t)(cValues), (C.size_t)(cSize), (C._Bool)(isInto), (*C.VecOpsConfig)(cCfg), (*C.g2_projective_t)(cValues))
	err := runtime.EIcicleError(cErr)
	return err
}

func G2ProjectiveToMontgomery(points core.HostOrDeviceSlice) runtime.EIcicleError {
	return convertG2ProjectivePointsMontgomery(points, true)
}

func G2ProjectiveFromMontgomery(points core.HostOrDeviceSlice) runtime.EIcicleError {
	return convertG2ProjectivePointsMontgomery(points, false)
}
