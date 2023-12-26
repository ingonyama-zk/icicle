package bn254

// #cgo CFLAGS: -I./include/
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254
// #include "curve.h"
import "C"

import (
	core "local/hello/icicle/wrappers/golang/core"
	cr "local/hello/icicle/wrappers/golang/cuda_runtime"
	"unsafe"
)

type Projective struct {
	core.Projective
}

func NewBN254Projective() Projective {
	return Projective{
		core.Projective{
			X: newBN254Field(),
			Y: newBN254Field(),
			Z: newBN254Field(),
		},
	}
}

// When templatizing, move this to be its own curveG2.go with the same template 
// as curve.go.tmpl but with g2bn254 as curve name
// func NewG2BN254Projective() Projective {
// 	return Projective{
// 		core.Projective{
// 			X: newG2BN254Field(),
// 			Y: newG2BN254Field(),
// 			Z: newG2BN254Field(),
// 		},
// 	}
// }

func (p *Projective) Eq(p2 *Projective) bool {
	cP := (*C.projective_t)(unsafe.Pointer(p))
	cP2 := (*C.projective_t)(unsafe.Pointer(p2))
	__ret := C.bn254Eq(cP, cP2)
	return __ret == (C._Bool)(true)
}

func (p *Projective) ToAffine() (a Affine) {
	cA := (*C.affine_t)(unsafe.Pointer(&a))
	cP := (*C.projective_t)(unsafe.Pointer(p))
	C.bn254ToAffine(cP, cA)
	return a
}

func GenerateProjectivePoints(size int) []Projective {
	points := make([]Projective, size)
	pPoints := (*C.projective_t)(unsafe.Pointer(&points[0]))
	cSize := (C.int)(size)
	C.bn254GenerateProjectivePoints(pPoints, cSize)

	return points
}

type Affine struct {
	core.Affine
}

func NewBN254Affine() Affine {
	return Affine {
		core.Affine{
			X: newBN254Field(),
			Y: newBN254Field(),
		},
	}
}

func (a* Affine) FromProjective(p *Projective) {
	aff := p.ToAffine()
	a.X = aff.X
	a.Y = aff.Y
}

func GenerateAffinePoints(size int) []Affine {
	points := make([]Affine, size)
	cPoints := (*C.affine_t)(unsafe.Pointer(&points[0]))
	cSize := (C.int)(size)
	C.bn254GenerateAffinePoints(cPoints, cSize)

	return points
}

func convertAffinePointsMontgomery(points cr.HostOrDeviceSlice[any, any], isInto bool) cr.CudaError {
	cValues := (*C.affine_t)(unsafe.Pointer(points.AsPointer()))
	cSize := (C.size_t)(points.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bn254AffineConvertMontgomery(cValues, cSize, cIsInto, cCtx) // templatize this per curve??
	err := (cr.CudaError)(__ret)
	return err
}

func ConvertAffineToMontgomery(points cr.HostOrDeviceSlice[any, any]) cr.CudaError {
	return convertAffinePointsMontgomery(points, true)
}

func ConvertAffineFromMontgomery(points cr.HostOrDeviceSlice[any, any]) cr.CudaError {
	return convertAffinePointsMontgomery(points, false)
}

func convertProjectivePointsMontgomery(points cr.HostOrDeviceSlice[any, any], isInto bool) cr.CudaError {
	cValues := (*C.projective_t)(unsafe.Pointer(points.AsPointer()))
	cSize := (C.size_t)(points.Len())
	cIsInto := (C._Bool)(isInto)
	defaultCtx, _ := cr.GetDefaultDeviceContext()
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&defaultCtx))
	__ret := C.bn254ProjectiveConvertMontgomery(cValues, cSize, cIsInto, cCtx) // templatize this per curve??
	err := (cr.CudaError)(__ret)
	return err
}

func ConvertProjectiveToMontgomery(points cr.HostOrDeviceSlice[any, any]) cr.CudaError {
	return convertProjectivePointsMontgomery(points, true)
}

func ConvertProjectiveFromMontgomery(points cr.HostOrDeviceSlice[any, any]) cr.CudaError {
	return convertProjectivePointsMontgomery(points, false)
}
