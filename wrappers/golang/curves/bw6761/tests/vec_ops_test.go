package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bw6_761 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bw6761/vecOps"
	"github.com/stretchr/testify/suite"
)

func testBw6_761VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := bw6_761.GenerateScalars(testSize)
	b := bw6_761.GenerateScalars(testSize)
	var scalar bw6_761.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bw6_761.ScalarField], testSize)
	out2 := make(core.HostSlice[bw6_761.ScalarField], testSize)
	out3 := make(core.HostSlice[bw6_761.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testBw6_761Transpose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := bw6_761.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[bw6_761.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[bw6_761.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[bw6_761.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testBw6_761Sum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := bw6_761.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[bw6_761.ScalarField], batchSize)
	result2 := make(core.HostSlice[bw6_761.ScalarField], batchSize)

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

func testBw6_761Product(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := bw6_761.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[bw6_761.ScalarField], batchSize)
	result2 := make(core.HostSlice[bw6_761.ScalarField], batchSize)

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

type Bw6_761VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bw6_761VecOpsTestSuite) TestBw6_761VecOps() {
	s.Run("TestBw6_761VecOps", testWrapper(&s.Suite, testBw6_761VecOps))
	s.Run("TestBw6_761Transpose", testWrapper(&s.Suite, testBw6_761Transpose))
	s.Run("TestBw6_761Sum", testWrapper(&s.Suite, testBw6_761Sum))
	s.Run("TestBw6_761Product", testWrapper(&s.Suite, testBw6_761Product))

}

func TestSuiteBw6_761VecOps(t *testing.T) {
	suite.Run(t, new(Bw6_761VecOpsTestSuite))
}
