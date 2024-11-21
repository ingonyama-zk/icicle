package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/polynomial"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/vecOps"
	"github.com/stretchr/testify/suite"
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

func testPolyCreateFromCoefficients(suite *suite.Suite) {
	scalars := bls12_377.GenerateScalars(33)
	var uniPoly polynomial.DensePolynomial

	poly := uniPoly.CreateFromCoeffecitients(scalars)
	poly.Print()
}

func testPolyEval(suite *suite.Suite) {
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
	suite.Equal(expected.FromUint32(277), fEvaled.(core.HostSlice[bls12_377.ScalarField])[0])
}

func testPolyClone(suite *suite.Suite) {
	f := randomPoly(8)
	x := rand()
	fx := f.Eval(x)

	g := f.Clone()
	fg := f.Add(&g)

	gx := g.Eval(x)
	fgx := fg.Eval(x)

	suite.Equal(fx, gx)
	suite.Equal(vecOp(fx, gx, core.Add), fgx)
}

func testPolyAddSubMul(suite *suite.Suite) {
	testSize := 1 << 10
	f := randomPoly(testSize)
	g := randomPoly(testSize)
	x := rand()

	fx := f.Eval(x)
	gx := g.Eval(x)

	polyAdd := f.Add(&g)
	fxAddgx := vecOp(fx, gx, core.Add)
	suite.Equal(polyAdd.Eval(x), fxAddgx)

	polySub := f.Subtract(&g)
	fxSubgx := vecOp(fx, gx, core.Sub)
	suite.Equal(polySub.Eval(x), fxSubgx)

	polyMul := f.Multiply(&g)
	fxMulgx := vecOp(fx, gx, core.Mul)
	suite.Equal(polyMul.Eval(x), fxMulgx)

	s1 := rand()
	polMulS1 := f.MultiplyByScalar(s1)
	suite.Equal(polMulS1.Eval(x), vecOp(fx, s1, core.Mul))

	s2 := rand()
	polMulS2 := f.MultiplyByScalar(s2)
	suite.Equal(polMulS2.Eval(x), vecOp(fx, s2, core.Mul))
}

func testPolyMonomials(suite *suite.Suite) {
	var zero bls12_377.ScalarField
	var f polynomial.DensePolynomial
	f.CreateFromCoeffecitients(core.HostSliceFromElements([]bls12_377.ScalarField{one, zero, two}))
	x := rand()

	fx := f.Eval(x)
	f.AddMonomial(three, 1)
	fxAdded := f.Eval(x)
	suite.Equal(fxAdded, vecOp(fx, vecOp(three, x, core.Mul), core.Add))

	f.SubMonomial(one, 0)
	fxSub := f.Eval(x)
	suite.Equal(fxSub, vecOp(fxAdded, one, core.Sub))
}

func testPolyReadCoeffs(suite *suite.Suite) {
	var f polynomial.DensePolynomial
	coeffs := core.HostSliceFromElements([]bls12_377.ScalarField{one, two, three, four})
	f.CreateFromCoeffecitients(coeffs)
	coeffsCopied := make(core.HostSlice[bls12_377.ScalarField], coeffs.Len())
	_, _ = f.CopyCoeffsRange(0, coeffs.Len()-1, coeffsCopied)
	suite.ElementsMatch(coeffs, coeffsCopied)

	var coeffsDevice core.DeviceSlice
	coeffsDevice.Malloc(one.Size(), coeffs.Len())
	_, _ = f.CopyCoeffsRange(0, coeffs.Len()-1, coeffsDevice)
	coeffsHost := make(core.HostSlice[bls12_377.ScalarField], coeffs.Len())
	coeffsHost.CopyFromDevice(&coeffsDevice)

	suite.ElementsMatch(coeffs, coeffsHost)
}

func testPolyOddEvenSlicing(suite *suite.Suite) {
	size := 1<<10 - 3
	f := randomPoly(size)

	even := f.Even()
	odd := f.Odd()
	suite.Equal(f.Degree(), even.Degree()+odd.Degree()+1)

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
	suite.Equal(evenExpected, evenEvaled)

	oddEvaled := odd.Eval(x)
	suite.Equal(oddExpected, oddEvaled)
}

func testPolynomialDivision(suite *suite.Suite) {
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
	suite.Equal(fEval, fReconEval)
}

func testDivideByVanishing(suite *suite.Suite) {
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
	suite.Equal(fDegree+4, fvDegree)

	fReconstructed := fv.DivideByVanishing(4)
	suite.Equal(fDegree, fReconstructed.Degree())

	x := rand()
	suite.Equal(f.Eval(x), fReconstructed.Eval(x))
}

// func TestPolySlice(suite *suite.Suite) {
// 	size := 4
// 	coeffs := bls12_377.GenerateScalars(size)
// 	var f DensePolynomial
// 	f.CreateFromCoeffecitients(coeffs)
// 	fSlice := f.AsSlice()
// 	suite.True(fSlice.IsOnDevice())
// 	suite.Equal(size, fSlice.Len())

// 	hostSlice := make(core.HostSlice[bls12_377.ScalarField], size)
// 	hostSlice.CopyFromDevice(fSlice)
// 	suite.Equal(coeffs, hostSlice)

// 	cfg := ntt.GetDefaultNttConfig()
// 	res := make(core.HostSlice[bls12_377.ScalarField], size)
// 	ntt.Ntt(fSlice, core.KForward, cfg, res)

// 	suite.Equal(f.Eval(one), res[0])
// }

type PolynomialTestSuite struct {
	suite.Suite
}

func (s *PolynomialTestSuite) TestPolynomial() {
	s.Run("TestPolyCreateFromCoefficients", testWrapper(&s.Suite, testPolyCreateFromCoefficients))
	s.Run("TestPolyEval", testWrapper(&s.Suite, testPolyEval))
	s.Run("TestPolyClone", testWrapper(&s.Suite, testPolyClone))
	s.Run("TestPolyAddSubMul", testWrapper(&s.Suite, testPolyAddSubMul))
	s.Run("TestPolyMonomials", testWrapper(&s.Suite, testPolyMonomials))
	s.Run("TestPolyReadCoeffs", testWrapper(&s.Suite, testPolyReadCoeffs))
	s.Run("TestPolyOddEvenSlicing", testWrapper(&s.Suite, testPolyOddEvenSlicing))
	s.Run("TestPolynomialDivision", testWrapper(&s.Suite, testPolynomialDivision))
	s.Run("TestDivideByVanishing", testWrapper(&s.Suite, testDivideByVanishing))
}

func TestSuitePolynomial(t *testing.T) {
	suite.Run(t, new(PolynomialTestSuite))
}
