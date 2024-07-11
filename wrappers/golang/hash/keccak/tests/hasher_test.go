package tests

import (
	"encoding/hex"
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/hash/keccak"

	"github.com/stretchr/testify/assert"
)

func createHostSliceFromHexString(hexString string) core.HostSlice[uint8] {
	byteArray, err := hex.DecodeString(hexString)
	if err != nil {
		panic("Not a hex string")
	}
	return core.HostSliceFromElements([]uint8(byteArray))
}

func TestSimpleHash256(t *testing.T) {
	input := createHostSliceFromHexString("1725b6679911bfe75ad7e248fbeec4a01034feace99aca43cd95d338a37db8d04b4aa5d83c8f8f5bdb8f7f98cec9a658f7f8061a6de07dcfd61db169cc7e666e1971adb4c7e97c43361c9a9eed8bb7b5c33cfe576a383a7440803996fd28148b")
	outHost := make(core.HostSlice[uint8], 32)

	cfg := keccak.GetDefaultHashConfig()
	e := keccak.Keccak256(input, int32(input.Len()), 1, outHost, &cfg)
	assert.Equal(t, e.CudaErrorCode, cr.CudaSuccess, "Hashing failed")
	t.Log(outHost)
	assert.Equal(t, outHost, createHostSliceFromHexString("10fd4a3df6046e32f282cad3ac78e1566304339e7a6696826af023a55ab42048"))
}

func TestBatchHash256(t *testing.T) {
	input := createHostSliceFromHexString("1725b6679911bfe75ad7e248fbeec4a01034feace99aca43cd95d338a37db8d04b4aa5d83c8f8f5bdb8f7f98cec9a658f7f8061a6de07dcfd61db169cc7e666e1971adb4c7e97c43361c9a9eed8bb7b5c33cfe576a383a7440803996fd28148b")
	outHost := make(core.HostSlice[uint8], 32*2)

	cfg := keccak.GetDefaultHashConfig()
	e := keccak.Keccak256(input, int32(input.Len()/2), 2, outHost, &cfg)
	assert.Equal(t, e.CudaErrorCode, cr.CudaSuccess, "Hashing failed")
	t.Log(outHost)
	assert.Equal(t, outHost[:32], createHostSliceFromHexString("7983fbc4cb4539cc90731205c44f74ca74e0a49ad1032a7a1429b1e443e66f45"))
	assert.Equal(t, outHost[32:64], createHostSliceFromHexString("2952c2491c75338d28943231a492e9ab684a6820e4af1d74c8c1976759f7bf4b"))
}

func TestSimpleHash512(t *testing.T) {
	input := createHostSliceFromHexString("1725b6679911bfe75ad7e248fbeec4a01034feace99aca43cd95d338a37db8d04b4aa5d83c8f8f5bdb8f7f98cec9a658f7f8061a6de07dcfd61db169cc7e666e1971adb4c7e97c43361c9a9eed8bb7b5c33cfe576a383a7440803996fd28148b")
	outHost := make(core.HostSlice[uint8], 64)

	cfg := keccak.GetDefaultHashConfig()
	e := keccak.Keccak512(input, int32(input.Len()), 1, outHost, &cfg)
	assert.Equal(t, e.CudaErrorCode, cr.CudaSuccess, "Hashing failed")
	t.Log(outHost)
	assert.Equal(t, outHost, createHostSliceFromHexString("1da4e0264dc755bc0b3a3318d2496e11c72322104693b68dbddfa66aa6e8b95526e95a7684a55ea831202f475f3d6a322ed86360d7e0e80f4a129f15d59dd403"))
}

func TestBatchHash512(t *testing.T) {
	input := createHostSliceFromHexString("1725b6679911bfe75ad7e248fbeec4a01034feace99aca43cd95d338a37db8d04b4aa5d83c8f8f5bdb8f7f98cec9a658f7f8061a6de07dcfd61db169cc7e666e1971adb4c7e97c43361c9a9eed8bb7b5c33cfe576a383a7440803996fd28148b")
	outHost := make(core.HostSlice[uint8], 64*2)

	cfg := keccak.GetDefaultHashConfig()
	e := keccak.Keccak512(input, int32(input.Len()/2), 2, outHost, &cfg)
	assert.Equal(t, e.CudaErrorCode, cr.CudaSuccess, "Hashing failed")
	t.Log(outHost)
	assert.Equal(t, outHost[:64], createHostSliceFromHexString("709974f0dc1df1461fcbc2275e968fcb510c947d38837d577d661b6b40249c6b348e33092e4795faad7d2829403bd70fe860207f40a84a23e03c4610ca7927a9"))
	assert.Equal(t, outHost[64:128], createHostSliceFromHexString("b8e46caa6cf7fbe6858deb28d4d9e58b768333b1260f5386656c0ae0d0850262bf6aa00293ef0979c37903fb5d2b784a02a4a227725a2b091df182abda03231d"))
}
