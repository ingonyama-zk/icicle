package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/vecOps"
	"github.com/stretchr/testify/suite"
)

func testBls12_377VecOps(suite suite.Suite) {
	testSize := 1 << 14

	a := bls12_377.GenerateScalars(testSize)
	b := bls12_377.GenerateScalars(testSize)
	var scalar bls12_377.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out2 := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out3 := make(core.HostSlice[bls12_377.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testBls12_377Transpose(suite suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := bls12_377.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

type Bls12_377VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bls12_377VecOpsTestSuite) TestBls12_377VecOps() {
	s.Run("TestBls12_377VecOps", testWrapper(s.Suite, testBls12_377VecOps))
	s.Run("TestBls12_377Transpose", testWrapper(s.Suite, testBls12_377Transpose))
}

func TestSuiteBls12_377VecOps(t *testing.T) {
	suite.Run(t, new(Bls12_377VecOpsTestSuite))
}
