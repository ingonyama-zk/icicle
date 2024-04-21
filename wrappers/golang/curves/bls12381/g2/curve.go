package g2

// #cgo CFLAGS: -I./include/
// #include "curve.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

type _g2Projective struct {
	X, Y, Z _g2BaseField
}

func (p _g2Projective) Size() int {
	return p.X.Size() * 3
}

func (p _g2Projective) AsPointer() *uint32 {
	return p.X.AsPointer()
}

func (p *_g2Projective) Zero() _g2Projective {
	p.X.Zero()
	p.Y.One()
	p.Z.Zero()

	return *p
}

func (p *_g2Projective) FromLimbs(x, y, z []uint32) _g2Projective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p *_g2Projective) FromAffine(a _g2Affine) _g2Projective {
	z := _g2BaseField{}
	z.One()

	p.X = a.X
	p.Y = a.Y
	p.Z = z

	return *p
}

func (p _g2Projective) ProjectiveEq(p2 *_g2Projective) bool {
	cP := (*C._g2_projective_t)(unsafe.Pointer(&p))
	cP2 := (*C._g2_projective_t)(unsafe.Pointer(&p2))
	__ret := C.bls12_381_g2_eq(cP, cP2)
	return __ret == (C._Bool)(true)
}

func (p *_g2Projective) ProjectiveToAffine() _g2Affine {
	var a _g2Affine

	cA := (*C._g2_affine_t)(unsafe.Pointer(&a))
	cP := (*C._g2_projective_t)(unsafe.Pointer(&p))
	C.bls12_381_g2_to_affine(cP, cA)
	return a
}

func _g2GenerateProjectivePoints(size int) core.HostSlice[_g2Projective] {
	points := make([]_g2Projective, size)
	for i := range points {
		points[i] = _g2Projective{}
	}

	pointsSlice := core.HostSliceFromElements[_g2Projective](points)
	pPoints := (*C._g2_projective_t)(unsafe.Pointer(&pointsSlice[0]))
	cSize := (C.int)(size)
	C.bls12_381_g2_generate_projective_points(pPoints, cSize)

	return pointsSlice
}

type _g2Affine struct {
	X, Y _g2BaseField
}

func (a _g2Affine) Size() int {
	return a.X.Size() * 2
}

func (a _g2Affine) AsPointer() *uint32 {
	return a.X.AsPointer()
}

func (a *_g2Affine) Zero() _g2Affine {
	a.X.Zero()
	a.Y.Zero()

	return *a
}

func (a *_g2Affine) FromLimbs(x, y []uint32) _g2Affine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a _g2Affine) ToProjective() _g2Projective {
	var z _g2BaseField

	return _g2Projective{
		X: a.X,
		Y: a.Y,
		Z: z.One(),
	}
}

func _g2AffineFromProjective(p *_g2Projective) _g2Affine {
	return p.ProjectiveToAffine()
}

func _g2GenerateAffinePoints(size int) core.HostSlice[_g2Affine] {
	points := make([]_g2Affine, size)
	for i := range points {
		points[i] = _g2Affine{}
	}

	pointsSlice := core.HostSliceFromElements[_g2Affine](points)
	cPoints := (*C._g2_affine_t)(unsafe.Pointer(&pointsSlice[0]))
	cSize := (C.int)(size)
	C.bls12_381_g2_generate_affine_points(cPoints, cSize)

	return pointsSlice
}

func convert_g2AffinePointsMontgomery(points *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C._g2_affine_t)(points.AsUnsafePointer())
	cSize := (C.size_t)(points.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bls12_381_g2_affine_convert_montgomery(cValues, cSize, cIsInto, cCtx)
	err := (cr.CudaError)(__ret)
	return err
}

func _g2AffineToMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convert_g2AffinePointsMontgomery(points, true)
}

func _g2AffineFromMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convert_g2AffinePointsMontgomery(points, false)
}

func convert_g2ProjectivePointsMontgomery(points *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C._g2_projective_t)(points.AsUnsafePointer())
	cSize := (C.size_t)(points.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bls12_381_g2_projective_convert_montgomery(cValues, cSize, cIsInto, cCtx)
	err := (cr.CudaError)(__ret)
	return err
}

func _g2ProjectiveToMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convert_g2ProjectivePointsMontgomery(points, true)
}

func _g2ProjectiveFromMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convert_g2ProjectivePointsMontgomery(points, false)
}
