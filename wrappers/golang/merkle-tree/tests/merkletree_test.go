package tests

import (
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	merkletree "github.com/ingonyama-zk/icicle/v3/wrappers/golang/merkle-tree"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testMerkleTree(s *suite.Suite) {
	numLayers := 4
	numElems := 1 << numLayers
	leafElemSize := 32
	keccak256, err := hash.NewKeccak256Hasher(uint64(2 * leafElemSize))
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("TestMerkleTree: Could not create keccak hasher due to: %v", err))
	}

	hashers := make([]hash.Hasher, numLayers)
	for i := 0; i < numLayers; i++ {
		hashers[i] = keccak256
	}

	mt, err := merkletree.CreateMerkleTree(hashers, uint64(leafElemSize), 0)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("TestMerkleTree: Could not create merkle tree due to: %v", err.AsString()))
	}

	leaves := make([]byte, numElems*leafElemSize)
	rand.Read(leaves)

	merkletree.BuildMerkleTree[byte](&mt, core.HostSliceFromElements(leaves), core.GetDefaultMerkleTreeConfig())

	mp, _ := merkletree.GetMerkleTreeProof[byte](
		&mt,
		core.HostSliceFromElements(leaves),
		1,     /* leafIndex */
		false, /* prunedPath */
		core.GetDefaultMerkleTreeConfig(),
	)

	root := merkletree.GetMerkleProofRoot[byte](&mp)
	path := merkletree.GetMerkleProofPath[byte](&mp)
	leaf, leafIndex := merkletree.GetMerkleProofLeaf[byte](&mp)

	fmt.Println("MerkleProof root:", root)
	fmt.Println("MerkleProof path for leaf index", leafIndex, ":", path)
	fmt.Println("MerkleProof leaf:", leaf)

	isVerified, err := mt.Verify(&mp)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("TestMerkleTree: Could not verify merkle tree due to: %v", err))
	}

	runtime.SetDevice(&devices[1])
	d_keccak256, err := hash.NewKeccak256Hasher(uint64(2 * leafElemSize))
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("TestMerkleTree: Could not create keccak hasher due to: %v", err))
	}

	d_hashers := make([]hash.Hasher, numLayers)
	for i := 0; i < numLayers; i++ {
		d_hashers[i] = d_keccak256
	}

	d_mt, err := merkletree.CreateMerkleTree(hashers, uint64(leafElemSize), 0)
	if err != runtime.Success {
		s.FailNow(fmt.Sprintf("TestMerkleTree: Could not create merkle tree due to: %v", err.AsString()))
	}

	merkletree.BuildMerkleTree[byte](&d_mt, core.HostSliceFromElements(leaves), core.GetDefaultMerkleTreeConfig())

	d_mp, _ := merkletree.GetMerkleTreeProof[byte](
		&d_mt,
		core.HostSliceFromElements(leaves),
		1,     /* leafIndex */
		false, /* prunedPath */
		core.GetDefaultMerkleTreeConfig(),
	)

	d_root := merkletree.GetMerkleProofRoot[byte](&d_mp)
	d_path := merkletree.GetMerkleProofPath[byte](&d_mp)
	d_leaf, d_leafIndex := merkletree.GetMerkleProofLeaf[byte](&d_mp)

	s.Equal(root, d_root)
	s.Equal(path, d_path)
	s.Equal(leaf, d_leaf)
	s.Equal(leafIndex, d_leafIndex)

	s.True(isVerified)
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
