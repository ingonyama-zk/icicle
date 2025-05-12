package polynomial

// #cgo CFLAGS: -I./include/
// #include "polynomial.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	goldilocks "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/goldilocks"
)

type PolynomialHandle = C.struct_PolynomialInst

type DensePolynomial struct {
	handle *PolynomialHandle
}

func (up *DensePolynomial) Print() {
	C.goldilocks_polynomial_print(up.handle)
}

func (up *DensePolynomial) CreateFromCoeffecitients(coeffs core.HostOrDeviceSlice) DensePolynomial {
	if coeffs.IsOnDevice() {
		coeffs.(core.DeviceSlice).CheckDevice()
	}
	coeffsPointer := (*C.scalar_t)(coeffs.AsUnsafePointer())
	cSize := (C.size_t)(coeffs.Len())
	up.handle = C.goldilocks_polynomial_create_from_coefficients(coeffsPointer, cSize)
	return *up
}

func (up *DensePolynomial) CreateFromROUEvaluations(evals core.HostOrDeviceSlice) DensePolynomial {
	evalsPointer := (*C.scalar_t)(evals.AsUnsafePointer())
	cSize := (C.size_t)(evals.Len())
	up.handle = C.goldilocks_polynomial_create_from_coefficients(evalsPointer, cSize)
	return *up
}

func (up *DensePolynomial) Clone() DensePolynomial {
	return DensePolynomial{
		handle: C.goldilocks_polynomial_clone(up.handle),
	}
}

func (up *DensePolynomial) Delete() {
	C.goldilocks_polynomial_delete(up.handle)
}

func (up *DensePolynomial) Add(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.goldilocks_polynomial_add(up.handle, b.handle),
	}
}

func (up *DensePolynomial) AddInplace(b *DensePolynomial) {
	C.goldilocks_polynomial_add_inplace(up.handle, b.handle)
}

func (up *DensePolynomial) Subtract(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.goldilocks_polynomial_subtract(up.handle, b.handle),
	}
}

func (up *DensePolynomial) Multiply(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.goldilocks_polynomial_multiply(up.handle, b.handle),
	}
}

func (up *DensePolynomial) MultiplyByScalar(scalar goldilocks.ScalarField) DensePolynomial {
	cScalar := (*C.scalar_t)(unsafe.Pointer(scalar.AsPointer()))
	return DensePolynomial{
		handle: C.goldilocks_polynomial_multiply_by_scalar(up.handle, cScalar),
	}
}

func (up *DensePolynomial) Divide(b *DensePolynomial) (DensePolynomial, DensePolynomial) {
	var q, r *PolynomialHandle
	C.goldilocks_polynomial_division(up.handle, b.handle, &q, &r)
	return DensePolynomial{
			handle: q,
		}, DensePolynomial{
			handle: r,
		}
}

func (up *DensePolynomial) Quotient(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.goldilocks_polynomial_quotient(up.handle, b.handle),
	}
}

func (up *DensePolynomial) Remainder(b *DensePolynomial) DensePolynomial {
	return DensePolynomial{
		handle: C.goldilocks_polynomial_remainder(up.handle, b.handle),
	}
}

func (up *DensePolynomial) DivideByVanishing(vanishing_degree uint64) DensePolynomial {
	cVanishingDegree := (C.ulong)(vanishing_degree)
	return DensePolynomial{
		handle: C.goldilocks_polynomial_divide_by_vanishing(up.handle, cVanishingDegree),
	}
}

func (up *DensePolynomial) AddMonomial(monomialCoeff goldilocks.ScalarField, monomial uint64) DensePolynomial {
	hs := core.HostSliceFromElements([]goldilocks.ScalarField{monomialCoeff})
	cMonomialCoeff := (*C.scalar_t)(hs.AsUnsafePointer())
	cMonomial := (C.ulong)(monomial)
	C.goldilocks_polynomial_add_monomial_inplace(up.handle, cMonomialCoeff, cMonomial)
	return *up
}

func (up *DensePolynomial) SubMonomial(monomialCoeff goldilocks.ScalarField, monomial uint64) DensePolynomial {
	hs := core.HostSliceFromElements([]goldilocks.ScalarField{monomialCoeff})
	cMonomialCoeff := (*C.scalar_t)(hs.AsUnsafePointer())
	cMonomial := (C.ulong)(monomial)
	C.goldilocks_polynomial_sub_monomial_inplace(up.handle, cMonomialCoeff, cMonomial)
	return *up
}

func (up *DensePolynomial) Eval(x goldilocks.ScalarField) goldilocks.ScalarField {
	domains := make(core.HostSlice[goldilocks.ScalarField], 1)
	domains[0] = x
	evals := make(core.HostSlice[goldilocks.ScalarField], 1)
	up.EvalOnDomain(domains, evals)
	return evals[0]
}

func (up *DensePolynomial) EvalOnDomain(domain, evals core.HostOrDeviceSlice) core.HostOrDeviceSlice {
	cDomain := (*C.scalar_t)(domain.AsUnsafePointer())
	cDomainSize := (C.size_t)(domain.Len())
	cEvals := (*C.scalar_t)(evals.AsUnsafePointer())
	C.goldilocks_polynomial_evaluate_on_domain(up.handle, cDomain, cDomainSize, cEvals)
	return evals
}

func (up *DensePolynomial) Degree() int {
	return int(C.goldilocks_polynomial_degree(up.handle))
}

func (up *DensePolynomial) CopyCoeffsRange(start, end int, out core.HostOrDeviceSlice) (int, core.HostOrDeviceSlice) {
	cStart := (C.size_t)(start)
	cEnd := (C.size_t)(end)
	cScalarOut := (*C.scalar_t)(out.AsUnsafePointer())
	__cNumCoeffsRead := C.goldilocks_polynomial_copy_coeffs_range(up.handle, cScalarOut, cStart, cEnd)
	return int(__cNumCoeffsRead), out
}

func (up *DensePolynomial) GetCoeff(idx int) goldilocks.ScalarField {
	out := make(core.HostSlice[goldilocks.ScalarField], 1)
	up.CopyCoeffsRange(idx, idx, out)
	return out[0]
}

func (up *DensePolynomial) Even() DensePolynomial {
	evenPoly := C.goldilocks_polynomial_even(up.handle)
	return DensePolynomial{
		handle: evenPoly,
	}
}

func (up *DensePolynomial) Odd() DensePolynomial {
	oddPoly := C.goldilocks_polynomial_odd(up.handle)
	return DensePolynomial{
		handle: oddPoly,
	}
}
