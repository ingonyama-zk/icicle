package tests

import (
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testKeccakBatch(s *suite.Suite) {
	single_hash_input_size := 30
	batch := 3
	const outputBytes = 64 // 64 bytes is output size of Keccak512

	input := make([]byte, single_hash_input_size*batch)
	_, err := rand.Read(input)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	keccakHasher, error := hash.NewKeccak512Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes*batch)
	keccakHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	runtime.SetDevice(&devices[1])
	keccakHasher, error = hash.NewKeccak512Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputMain := make([]byte, outputBytes*batch)
	keccakHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputMain),
		core.GetDefaultHashConfig(),
	)

	outputEmpty := make([]byte, outputBytes*batch)
	s.Equal(outputRef, outputMain)
	s.NotEqual(outputEmpty, outputMain)
}

func testBlake2s(s *suite.Suite) {
	singleHashInputSize := 567
	batch := 11
	const outputBytes = 32 // 32 bytes is output size of Blake2s

	input := make([]byte, singleHashInputSize*batch)
	_, err := rand.Read(input)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	Blake2sHasher, error := hash.NewBlake2sHasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes*batch)
	Blake2sHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	runtime.SetDevice(&devices[1])
	Blake2sHasher, error = hash.NewBlake2sHasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputMain := make([]byte, outputBytes*batch)
	Blake2sHasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputMain),
		core.GetDefaultHashConfig(),
	)

	outputEmpty := make([]byte, outputBytes*batch)
	s.Equal(outputRef, outputMain)
	s.NotEqual(outputEmpty, outputMain)
}

func testBlake3(s *suite.Suite) {
	const outputBytes = 32 // 32 bytes is output size of Blake3

	// Known input string and expected hash
	inputString := "Hello world I am blake32"
	expectedHash := "cb46bdd080609257ba2cca93b21d1f72ff1737eb48790f3c17ceae83b6c74e42"

	input := []byte(inputString)

	Blake3Hasher, error := hash.NewBlake3Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes)
	Blake3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	outputRefHex := fmt.Sprintf("%x", outputRef)

	s.Equal(expectedHash, outputRefHex, "Hash mismatch: got %s, expected %s", outputRefHex, expectedHash)
}

func testSha3(s *suite.Suite) {
	singleHashInputSize := 1153
	batch := 1
	const outputBytes = 32 // 32 bytes is output size of Sha3 256

	input := make([]byte, singleHashInputSize*batch)
	_, err := rand.Read(input)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	Sha3Hasher, error := hash.NewSha3256Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputRef := make([]byte, outputBytes*batch)
	Sha3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputRef),
		core.GetDefaultHashConfig(),
	)

	runtime.SetDevice(&devices[1])
	Sha3Hasher, error = hash.NewSha3256Hasher(0 /*default chunk size*/)
	if error != runtime.Success {
		fmt.Println("error:", error)
		return
	}

	outputMain := make([]byte, outputBytes*batch)
	Sha3Hasher.Hash(
		core.HostSliceFromElements(input),
		core.HostSliceFromElements(outputMain),
		core.GetDefaultHashConfig(),
	)

	outputEmpty := make([]byte, outputBytes*batch)
	s.Equal(outputRef, outputMain)
	s.NotEqual(outputEmpty, outputMain)
}

type HashTestSuite struct {
	suite.Suite
}

func (s *HashTestSuite) TestHash() {
	s.Run("TestKeccakBatch", testWrapper(&s.Suite, testKeccakBatch))
	s.Run("TestBlake2s", testWrapper(&s.Suite, testBlake2s))
	s.Run("TestBlake3", testWrapper(&s.Suite, testBlake3))
	s.Run("TestSha3", testWrapper(&s.Suite, testSha3))
}

func TestSuiteHash(t *testing.T) {
	suite.Run(t, new(HashTestSuite))
}
