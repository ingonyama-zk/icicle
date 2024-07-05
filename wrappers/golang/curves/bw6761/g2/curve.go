package g2

// #cgo CFLAGS: -I./include/
// #include "curve.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
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
	C.bw6_761_g2_from_affine(cA, cP)
	return *p
}

func (p G2Projective) ProjectiveEq(p2 *G2Projective) bool {
	cP := (*C.g2_projective_t)(unsafe.Pointer(&p))
	cP2 := (*C.g2_projective_t)(unsafe.Pointer(&p2))
	__ret := C.bw6_761_g2_eq(cP, cP2)
	return __ret == (C._Bool)(true)
}

func (p *G2Projective) ProjectiveToAffine() G2Affine {
	var a G2Affine

	cA := (*C.g2_affine_t)(unsafe.Pointer(&a))
	cP := (*C.g2_projective_t)(unsafe.Pointer(p))
	C.bw6_761_g2_to_affine(cP, cA)
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
	C.bw6_761_g2_generate_projective_points(pPoints, cSize)

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

func (a *G2Affine) FromLimbs(x, y []uint32) G2Affine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a G2Affine) ToProjective() G2Projective {
	var p G2Projective

	cA := (*C.g2_affine_t)(unsafe.Pointer(&a))
	cP := (*C.g2_projective_t)(unsafe.Pointer(&p))
	C.bw6_761_g2_from_affine(cA, cP)
	return p
}

func G2AffineFromProjective(p *G2Projective) G2Affine {
	return p.ProjectiveToAffine()
}

func G2GenerateAffinePoints(size int) core.HostSlice[G2Affine] {
	points := make([]G2Affine, size)
	for i := range points {
		points[i] = G2Affine{}
	}

	pointsSlice := core.HostSliceFromElements[G2Affine](points)
	cPoints := (*C.g2_affine_t)(unsafe.Pointer(&pointsSlice[0]))
	cSize := (C.int)(size)
	C.bw6_761_g2_generate_affine_points(cPoints, cSize)

	return pointsSlice
}

func convertG2AffinePointsMontgomery(points *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C.g2_affine_t)(points.AsUnsafePointer())
	cSize := (C.size_t)(points.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bw6_761_g2_affine_convert_montgomery(cValues, cSize, cIsInto, cCtx)
	err := (cr.CudaError)(__ret)
	return err
}

func G2AffineToMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convertG2AffinePointsMontgomery(points, true)
}

func G2AffineFromMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convertG2AffinePointsMontgomery(points, false)
}

func convertG2ProjectivePointsMontgomery(points *core.DeviceSlice, isInto bool) cr.CudaError {
	cValues := (*C.g2_projective_t)(points.AsUnsafePointer())
	cSize := (C.size_t)(points.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bw6_761_g2_projective_convert_montgomery(cValues, cSize, cIsInto, cCtx)
	err := (cr.CudaError)(__ret)
	return err
}

func G2ProjectiveToMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convertG2ProjectivePointsMontgomery(points, true)
}

func G2ProjectiveFromMontgomery(points *core.DeviceSlice) cr.CudaError {
	points.CheckDevice()
	return convertG2ProjectivePointsMontgomery(points, false)
}
