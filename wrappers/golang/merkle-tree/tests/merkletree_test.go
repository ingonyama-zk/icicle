package tests

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

func testMerkleTree(s *suite.Suite) {
	
}

type MerkleTreeTestSuite struct {
	suite.Suite
}

func (s *MerkleTreeTestSuite) TestMerkleTree() {
	s.Run("TestMerkleTree", testWrapper(&s.Suite, testMerkleTree))
}

func TestSuiteMerkleTree(t *testing.T) {
	suite.Run(t, new(MerkleTreeTestSuite))
}