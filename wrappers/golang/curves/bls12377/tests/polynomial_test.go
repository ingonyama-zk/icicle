package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"

	// "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/ntt"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/polynomial"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/vecOps"
	"github.com/stretchr/testify/assert"
)

var one, two, three, four, five bls12_377.ScalarField

func init() {
	one.One()
	two.FromUint32(2)
	three.FromUint32(3)
	four.FromUint32(4)
	five.FromUint32(5)
}

func rand() bls12_377.ScalarField {
	return bls12_377.GenerateScalars(1)[0]
}

func randomPoly(size int) (f polynomial.DensePolynomial) {
	f.CreateFromCoeffecitients(core.HostSliceFromElements(bls12_377.GenerateScalars(size)))
	return f
}

func vecOp(a, b bls12_377.ScalarField, op core.VecOps) bls12_377.ScalarField {
	ahost := core.HostSliceWithValue(a, 1)
	bhost := core.HostSliceWithValue(b, 1)
	out := make(core.HostSlice[bls12_377.ScalarField], 1)

	cfg := core.DefaultVecOpsConfig()
	vecOps.VecOp(ahost, bhost, out, cfg, op)
	return out[0]
}

func TestPolyCreateFromCoefficients(t *testing.T) {
	scalars := bls12_377.GenerateScalars(33)
	var uniPoly polynomial.DensePolynomial

	poly := uniPoly.CreateFromCoeffecitients(scalars)
	poly.Print()
}

func TestPolyEval(t *testing.T) {
	// testing correct evaluation of f(8) for f(x)=4x^2+2x+5
	coeffs := core.HostSliceFromElements([]bls12_377.ScalarField{five, two, four})
	var f polynomial.DensePolynomial
	f.CreateFromCoeffecitients(coeffs)

	var x bls12_377.ScalarField
	x.FromUint32(8)
	domains := make(core.HostSlice[bls12_377.ScalarField], 1)
	domains[0] = x
	evals := make(core.HostSlice[bls12_377.ScalarField], 1)
	fEvaled := f.EvalOnDomain(domains, evals)
	var expected bls12_377.ScalarField
	assert.Equal(t, expected.FromUint32(277), fEvaled.(core.HostSlice[bls12_377.ScalarField])[0])
}

func TestPolyClone(t *testing.T) {
	f := randomPoly(8)
	x := rand()
	fx := f.Eval(x)

	g := f.Clone()
	fg := f.Add(&g)

	gx := g.Eval(x)
	fgx := fg.Eval(x)

	assert.Equal(t, fx, gx)
	assert.Equal(t, vecOp(fx, gx, core.Add), fgx)
}

func TestPolyAddSubMul(t *testing.T) {
	testSize := 1 << 10
	f := randomPoly(testSize)
	g := randomPoly(testSize)
	x := rand()

	fx := f.Eval(x)
	gx := g.Eval(x)

	polyAdd := f.Add(&g)
	fxAddgx := vecOp(fx, gx, core.Add)
	assert.Equal(t, polyAdd.Eval(x), fxAddgx)

	polySub := f.Subtract(&g)
	fxSubgx := vecOp(fx, gx, core.Sub)
	assert.Equal(t, polySub.Eval(x), fxSubgx)

	polyMul := f.Multiply(&g)
	fxMulgx := vecOp(fx, gx, core.Mul)
	assert.Equal(t, polyMul.Eval(x), fxMulgx)

	s1 := rand()
	polMulS1 := f.MultiplyByScalar(s1)
	assert.Equal(t, polMulS1.Eval(x), vecOp(fx, s1, core.Mul))

	s2 := rand()
	polMulS2 := f.MultiplyByScalar(s2)
	assert.Equal(t, polMulS2.Eval(x), vecOp(fx, s2, core.Mul))
}

func TestPolyMonomials(t *testing.T) {
	var zero bls12_377.ScalarField
	var f polynomial.DensePolynomial
	f.CreateFromCoeffecitients(core.HostSliceFromElements([]bls12_377.ScalarField{one, zero, two}))
	x := rand()

	fx := f.Eval(x)
	f.AddMonomial(three, 1)
	fxAdded := f.Eval(x)
	assert.Equal(t, fxAdded, vecOp(fx, vecOp(three, x, core.Mul), core.Add))

	f.SubMonomial(one, 0)
	fxSub := f.Eval(x)
	assert.Equal(t, fxSub, vecOp(fxAdded, one, core.Sub))
}

func TestPolyReadCoeffs(t *testing.T) {
	var f polynomial.DensePolynomial
	coeffs := core.HostSliceFromElements([]bls12_377.ScalarField{one, two, three, four})
	f.CreateFromCoeffecitients(coeffs)
	coeffsCopied := make(core.HostSlice[bls12_377.ScalarField], coeffs.Len())
	_, _ = f.CopyCoeffsRange(0, coeffs.Len()-1, coeffsCopied)
	assert.ElementsMatch(t, coeffs, coeffsCopied)

	var coeffsDevice core.DeviceSlice
	coeffsDevice.Malloc(coeffs.Len()*one.Size(), one.Size())
	_, _ = f.CopyCoeffsRange(0, coeffs.Len()-1, coeffsDevice)
	coeffsHost := make(core.HostSlice[bls12_377.ScalarField], coeffs.Len())
	coeffsHost.CopyFromDevice(&coeffsDevice)

	assert.ElementsMatch(t, coeffs, coeffsHost)
}

func TestPolyOddEvenSlicing(t *testing.T) {
	size := 1<<10 - 3
	f := randomPoly(size)

	even := f.Even()
	odd := f.Odd()
	assert.Equal(t, f.Degree(), even.Degree()+odd.Degree()+1)

	x := rand()
	var evenExpected, oddExpected bls12_377.ScalarField
	for i := size; i >= 0; i-- {
		if i%2 == 0 {
			mul := vecOp(evenExpected, x, core.Mul)
			evenExpected = vecOp(mul, f.GetCoeff(i), core.Add)
		} else {
			mul := vecOp(oddExpected, x, core.Mul)
			oddExpected = vecOp(mul, f.GetCoeff(i), core.Add)
		}
	}

	evenEvaled := even.Eval(x)
	assert.Equal(t, evenExpected, evenEvaled)

	oddEvaled := odd.Eval(x)
	assert.Equal(t, oddExpected, oddEvaled)
}

func TestPolynomialDivision(t *testing.T) {
	// divide f(x)/g(x), compute q(x), r(x) and check f(x)=q(x)*g(x)+r(x)
	var f, g polynomial.DensePolynomial
	f.CreateFromCoeffecitients(core.HostSliceFromElements(bls12_377.GenerateScalars(1 << 4)))
	g.CreateFromCoeffecitients(core.HostSliceFromElements(bls12_377.GenerateScalars(1 << 2)))

	q, r := f.Divide(&g)

	qMulG := q.Multiply(&g)
	fRecon := qMulG.Add(&r)

	x := bls12_377.GenerateScalars(1)[0]
	fEval := f.Eval(x)
	fReconEval := fRecon.Eval(x)
	assert.Equal(t, fEval, fReconEval)
}

func TestDivideByVanishing(t *testing.T) {
	// poly of x^4-1 vanishes ad 4th rou
	var zero bls12_377.ScalarField
	minus_one := vecOp(zero, one, core.Sub)
	coeffs := core.HostSliceFromElements([]bls12_377.ScalarField{minus_one, zero, zero, zero, one}) // x^4-1
	var v polynomial.DensePolynomial
	v.CreateFromCoeffecitients(coeffs)

	f := randomPoly(1 << 3)

	fv := f.Multiply(&v)
	fDegree := f.Degree()
	fvDegree := fv.Degree()
	assert.Equal(t, fDegree+4, fvDegree)

	fReconstructed := fv.DivideByVanishing(4)
	assert.Equal(t, fDegree, fReconstructed.Degree())

	x := rand()
	assert.Equal(t, f.Eval(x), fReconstructed.Eval(x))
}

// func TestPolySlice(t *testing.T) {
// 	size := 4
// 	coeffs := bls12_377.GenerateScalars(size)
// 	var f DensePolynomial
// 	f.CreateFromCoeffecitients(coeffs)
// 	fSlice := f.AsSlice()
// 	assert.True(t, fSlice.IsOnDevice())
// 	assert.Equal(t, size, fSlice.Len())

// 	hostSlice := make(core.HostSlice[bls12_377.ScalarField], size)
// 	hostSlice.CopyFromDevice(fSlice)
// 	assert.Equal(t, coeffs, hostSlice)

// 	cfg := ntt.GetDefaultNttConfig()
// 	res := make(core.HostSlice[bls12_377.ScalarField], size)
// 	ntt.Ntt(fSlice, core.KForward, cfg, res)

// 	assert.Equal(t, f.Eval(one), res[0])
// }
