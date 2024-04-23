package polynomial

// #cgo CFLAGS: -I./include/
// #include "polynomial.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254"
)

type PolynomialHandle = C.struct_PolynomialInst

type DensePolynomial struct {
	handle *PolynomialHandle
}

func InitPolyBackend() bool {
	return (bool)(C.bn254_polynomial_init_cuda_backend())
}

func (up *DensePolynomial) Print() {
	C.bn254_polynomial_print(up.handle)
}

func (up *DensePolynomial) CreateFromCoeffecitients(coeffs core.HostOrDeviceSlice) DensePolynomial {
	if coeffs.IsOnDevice() {
		coeffs.(core.DeviceSlice).CheckDevice()
	}
	coeffsPointer := (*C.scalar_t)(coeffs.AsUnsafePointer())
	cSize := (C.size_t)(coeffs.Len())
	up.handle = C.bn254_polynomial_create_from_coefficients(coeffsPointer, cSize)
	return *up
}

func (up *DensePolynomial) CreateFromROUEvaluations(evals core.HostOrDeviceSlice) DensePolynomial {
	evalsPointer := (*C.scalar_t)(evals.AsUnsafePointer())
	cSize := (C.size_t)(evals.Len())
	up.handle = C.bn254_polynomial_create_from_coefficients(evalsPointer, cSize)
	return *up
}

func (up *DensePolynomial) Clone() DensePolynomial {
	return DensePolynomial{
		handle: C.bn254_polynomial_clone(up.handle),
	}
}

// TODO @jeremyfelder: Maybe this should be in a SetFinalizer that is set on Create functions?
func (up *DensePolynomial) Delete() {
	C.bn254_polynomial_delete(up.handle)
}

func (up *DensePolynomial) Add(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.bn254_polynomial_add(up.handle, b.handle),
	}
}

func (up *DensePolynomial) AddInplace(b *DensePolynomial) {
	C.bn254_polynomial_add_inplace(up.handle, b.handle)
}

func (up *DensePolynomial) Subtract(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.bn254_polynomial_subtract(up.handle, b.handle),
	}
}

func (up *DensePolynomial) Multiply(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.bn254_polynomial_multiply(up.handle, b.handle),
	}
}

func (up *DensePolynomial) MultiplyByScalar(scalar bn254.ScalarField) DensePolynomial {
	cScalar := (*C.scalar_t)(unsafe.Pointer(scalar.AsPointer()))
	return DensePolynomial{
		handle: C.bn254_polynomial_multiply_by_scalar(up.handle, cScalar),
	}
}

func (up *DensePolynomial) Divide(b *DensePolynomial) (DensePolynomial, DensePolynomial) {
	var q, r *PolynomialHandle
	C.bn254_polynomial_division(up.handle, b.handle, &q, &r)
	return DensePolynomial{
			handle: q,
		}, DensePolynomial{
			handle: r,
		}
}

func (up *DensePolynomial) Quotient(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.bn254_polynomial_quotient(up.handle, b.handle),
	}
}

func (up *DensePolynomial) Remainder(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.bn254_polynomial_remainder(up.handle, b.handle),
	}
}

func (up *DensePolynomial) DivideByVanishing(vanishing_degree uint64) DensePolynomial {
	cVanishingDegree := (C.ulong)(vanishing_degree)
	return DensePolynomial{
		handle: C.bn254_polynomial_divide_by_vanishing(up.handle, cVanishingDegree),
	}
}

func (up *DensePolynomial) AddMonomial(monomialCoeff bn254.ScalarField, monomial uint64) DensePolynomial {
	hs := core.HostSliceFromElements([]bn254.ScalarField{monomialCoeff})
	cMonomialCoeff := (*C.scalar_t)(hs.AsUnsafePointer())
	cMonomial := (C.ulong)(monomial)
	C.bn254_polynomial_add_monomial_inplace(up.handle, cMonomialCoeff, cMonomial)
	return *up
}

func (up *DensePolynomial) SubMonomial(monomialCoeff bn254.ScalarField, monomial uint64) DensePolynomial {
	hs := core.HostSliceFromElements([]bn254.ScalarField{monomialCoeff})
	cMonomialCoeff := (*C.scalar_t)(hs.AsUnsafePointer())
	cMonomial := (C.ulong)(monomial)
	C.bn254_polynomial_sub_monomial_inplace(up.handle, cMonomialCoeff, cMonomial)
	return *up
}

func (up *DensePolynomial) Eval(x bn254.ScalarField) bn254.ScalarField {
	domains := make(core.HostSlice[bn254.ScalarField], 1)
	domains[0] = x
	evals := make(core.HostSlice[bn254.ScalarField], 1)
	up.EvalOnDomain(domains, evals)
	return evals[0]
}

func (up *DensePolynomial) EvalOnDomain(domain, evals core.HostOrDeviceSlice) core.HostOrDeviceSlice {
	cDomain := (*C.scalar_t)(domain.AsUnsafePointer())
	cDomainSize := (C.size_t)(domain.Len())
	cEvals := (*C.scalar_t)(evals.AsUnsafePointer())
	C.bn254_polynomial_evaluate_on_domain(up.handle, cDomain, cDomainSize, cEvals)
	return evals
}

func (up *DensePolynomial) Degree() int {
	return int(C.bn254_polynomial_degree(up.handle))
}

func (up *DensePolynomial) CopyCoeffsRange(start, end int, out core.HostOrDeviceSlice) (int, core.HostOrDeviceSlice) {
	cStart := (C.size_t)(start)
	cEnd := (C.size_t)(end)
	cScalarOut := (*C.scalar_t)(out.AsUnsafePointer())
	__cNumCoeffsRead := C.bn254_polynomial_copy_coeffs_range(up.handle, cScalarOut, cStart, cEnd)
	return int(__cNumCoeffsRead), out
}

func (up *DensePolynomial) GetCoeff(idx int) bn254.ScalarField {
	out := make(core.HostSlice[bn254.ScalarField], 1)
	up.CopyCoeffsRange(idx, idx, out)
	return out[0]
}

func (up *DensePolynomial) Even() DensePolynomial {
	evenPoly := C.bn254_polynomial_even(up.handle)
	return DensePolynomial{
		handle: evenPoly,
	}
}

func (up *DensePolynomial) Odd() DensePolynomial {
	oddPoly := C.bn254_polynomial_odd(up.handle)
	return DensePolynomial{
		handle: oddPoly,
	}
}
