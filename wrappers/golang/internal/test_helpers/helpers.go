package test_helpers

import (
	"math/rand"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

var (
	REFERENCE_DEVICE = runtime.CreateDevice("CPU", 0)
	MAIN_DEVICE      = runtime.CreateDevice("CUDA", 0)
)

func ActivateReferenceDevice() {
	runtime.SetDevice(&REFERENCE_DEVICE)
}

func ActivateMainDevice() {
	runtime.SetDevice(&MAIN_DEVICE)
}

func GenerateRandomLimb(size int) []uint32 {
	limbs := make([]uint32, size)
	for i := range limbs {
		limbs[i] = rand.Uint32()
	}
	return limbs
}

func GenerateLimbOne(size int) []uint32 {
	limbs := make([]uint32, size)
	limbs[0] = 1
	return limbs
}

func GenerateBytesArray(size int) ([]byte, []uint32) {
	baseBytes := []byte{1, 2, 3, 4}
	var bytes []byte
	var limbs []uint32
	for i := 0; i < size; i++ {
		bytes = append(bytes, baseBytes...)
		limbs = append(limbs, 67305985)
	}

	return bytes, limbs
}
