package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_381 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/vecOps"
	"github.com/stretchr/testify/suite"
)

func testBls12_381VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := bls12_381.GenerateScalars(testSize)
	b := bls12_381.GenerateScalars(testSize)
	var scalar bls12_381.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bls12_381.ScalarField], testSize)
	out2 := make(core.HostSlice[bls12_381.ScalarField], testSize)
	out3 := make(core.HostSlice[bls12_381.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testBls12_381Transpose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := bls12_381.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[bls12_381.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[bls12_381.ScalarField], rowSize*columnSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.TransposeMatrix(matrix, out, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(out, out2, rowSize, columnSize, cfg)

	suite.Equal(matrix, out2)

	var dMatrix, dOut, dOut2 core.DeviceSlice

	matrix.CopyToDevice(&dMatrix, true)
	dOut.Malloc(matrix.SizeOfElement(), columnSize*rowSize)
	dOut2.Malloc(matrix.SizeOfElement(), columnSize*rowSize)

	vecOps.TransposeMatrix(dMatrix, dOut, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(dOut, dOut2, rowSize, columnSize, cfg)
	output := make(core.HostSlice[bls12_381.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testBls12_381Sum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := bls12_381.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[bls12_381.ScalarField], batchSize)
	result2 := make(core.HostSlice[bls12_381.ScalarField], batchSize)

	cfg := core.DefaultVecOpsConfig()
	cfg.BatchSize = int32(batchSize)

	vecOps.SumScalars(a, result, cfg)

	// Test with device memory
	var dA, dResult core.DeviceSlice
	a.CopyToDevice(&dA, true)
	dResult.Malloc(a.SizeOfElement()*batchSize, batchSize)

	vecOps.SumScalars(dA, dResult, cfg)
	result2.CopyFromDevice(&dResult)

	suite.Equal(result, result2)
}

func testBls12_381Product(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := bls12_381.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[bls12_381.ScalarField], batchSize)
	result2 := make(core.HostSlice[bls12_381.ScalarField], batchSize)

	cfg := core.DefaultVecOpsConfig()
	cfg.BatchSize = int32(batchSize)

	vecOps.ProductScalars(a, result, cfg)

	// Test with device memory
	var dA, dResult core.DeviceSlice
	a.CopyToDevice(&dA, true)
	dResult.Malloc(a.SizeOfElement()*batchSize, batchSize)

	vecOps.ProductScalars(dA, dResult, cfg)
	result2.CopyFromDevice(&dResult)

	suite.Equal(result, result2)
}

type Bls12_381VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bls12_381VecOpsTestSuite) TestBls12_381VecOps() {
	s.Run("TestBls12_381VecOps", testWrapper(&s.Suite, testBls12_381VecOps))
	s.Run("TestBls12_381Transpose", testWrapper(&s.Suite, testBls12_381Transpose))
	s.Run("TestBls12_381Sum", testWrapper(&s.Suite, testBls12_381Sum))
	s.Run("TestBls12_381Product", testWrapper(&s.Suite, testBls12_381Product))

}

func TestSuiteBls12_381VecOps(t *testing.T) {
	suite.Run(t, new(Bls12_381VecOpsTestSuite))
}
