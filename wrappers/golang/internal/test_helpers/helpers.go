package test_helpers

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

var (
	LOAD_AND_INIT_DEVICES = sync.OnceFunc(func() {
		runtime.LoadBackendFromEnvOrDefault()
		registeredDevices, _ := runtime.GetRegisteredDevices()

		fmt.Println(registeredDevices)

		if len(registeredDevices) < 2 {
			panic("Tests require at least 2 backends to be loaded")
		}

		mainDeviceName := registeredDevices[1]
		if registeredDevices[0] != "CPU" {
			mainDeviceName = registeredDevices[0]
		}

		MAIN_DEVICE = runtime.CreateDevice(mainDeviceName, 0)
		fmt.Println("[INFO] Go testing: registeredDevices=", registeredDevices, "; MainDevice=", mainDeviceName, "; ReferenceDevice=CPU")
	})
	REFERENCE_DEVICE = runtime.CreateDevice("CPU", 0)
	MAIN_DEVICE      runtime.Device
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
