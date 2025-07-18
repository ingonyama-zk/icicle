package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"

	bls12377Ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"

	bn254Ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ntt"

	bls12377Fft "github.com/consensys/gnark-crypto/ecc/bls12-377/fr/fft"
	bn254Fft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func main() {
	var logSize int
	var deviceType string

	flag.IntVar(&logSize, "s", 20, "Log size")
	flag.StringVar(&deviceType, "device", "CUDA", "Device type")
	flag.Parse()

	if deviceType != "CPU" {
		runtime.LoadBackendFromEnvOrDefault()
	}

	device := runtime.CreateDevice(deviceType, 0)
	// NOTE: If you are only using a single device the entire time
	// 			then this is ok. If you are using multiple devices
	// 			then you should use runtime.RunOnDevice() instead.
	runtime.SetDefaultDevice(&device)

	size := 1 << logSize

	fmt.Printf("---------------------- NTT size 2^%d=%d ------------------------\n", logSize, size)

	print("Generating BN254 scalars ... ")
	startTime := time.Now()
	scalarsBn254 := bn254.GenerateScalars(size)
	println(time.Since(startTime).String())

	cfgBn254 := bn254Ntt.GetDefaultNttConfig()
	cfgBn254.IsAsync = true

	print("Generating BLS12_377 scalars ... ")
	startTime = time.Now()
	scalarsBls12377 := bls12377.GenerateScalars(size)
	println(time.Since(startTime).String())

	cfgBls12377 := bls12377Ntt.GetDefaultNttConfig()
	cfgBls12377.IsAsync = true
	cfgInitDomainBls := core.GetDefaultNTTInitDomainConfig()

	rouMontBn254, _ := bn254Fft.Generator(uint64(size))
	rouBn254 := rouMontBn254.Bits()
	rouIcicleBn254 := bn254.ScalarField{}
	limbsBn254 := core.ConvertUint64ArrToUint32Arr(rouBn254[:])
	rouIcicleBn254.FromLimbs(limbsBn254)
	bn254Ntt.InitDomain(rouIcicleBn254, cfgInitDomainBls)

	rouMontBls12377, _ := bls12377Fft.Generator(uint64(size))
	rouBls12377 := rouMontBls12377.Bits()
	rouIcicleBls12377 := bls12377.ScalarField{}
	limbsBls12377 := core.ConvertUint64ArrToUint32Arr(rouBls12377[:])
	rouIcicleBls12377.FromLimbs(limbsBls12377)
	bls12377Ntt.InitDomain(rouIcicleBls12377, cfgInitDomainBls)

	print("Configuring bn254 NTT ... ")
	startTime = time.Now()

	streamBn254, _ := runtime.CreateStream()

	cfgBn254.StreamHandle = streamBn254

	var nttResultBn254 core.DeviceSlice

	_, e := nttResultBn254.MallocAsync(scalarsBn254.SizeOfElement(), size, streamBn254)
	if e != runtime.Success {
		errorString := fmt.Sprint(
			"Bn254 Malloc failed: ", e)
		panic(errorString)
	}

	println(time.Since(startTime).String())

	print("Configuring Bls12377 NTT ... ")
	startTime = time.Now()

	streamBls12377, _ := runtime.CreateStream()

	cfgBls12377.StreamHandle = streamBls12377

	var nttResultBls12377 core.DeviceSlice

	_, e = nttResultBls12377.MallocAsync(scalarsBls12377.SizeOfElement(), size, streamBls12377)
	if e != runtime.Success {
		errorString := fmt.Sprint(
			"Bls12_377 Malloc failed: ", e)
		panic(errorString)
	}

	println(time.Since(startTime).String())

	print("Executing bn254 NTT on device ... ")
	startTime = time.Now()

	err := bn254Ntt.Ntt(scalarsBn254, core.KForward, &cfgBn254, nttResultBn254)
	if err != runtime.Success {
		errorString := fmt.Sprint(
			"bn254 Ntt failed: ", e)
		panic(errorString)
	}

	nttResultBn254Host := make(core.HostSlice[bn254.ScalarField], size)
	nttResultBn254Host.CopyFromDeviceAsync(&nttResultBn254, streamBn254)
	nttResultBn254.FreeAsync(streamBn254)
	runtime.SynchronizeStream(streamBn254)
	println(time.Since(startTime).String())

	print("Executing Bls12377 NTT on device ... ")
	startTime = time.Now()

	err = bls12377Ntt.Ntt(scalarsBls12377, core.KForward, &cfgBls12377, nttResultBls12377)
	if err != runtime.Success {
		errorString := fmt.Sprint(
			"bls12_377 Ntt failed: ", e)
		panic(errorString)
	}

	nttResultBls12377Host := make(core.HostSlice[bls12377.ScalarField], size)
	nttResultBls12377Host.CopyFromDeviceAsync(&nttResultBls12377, streamBls12377)
	nttResultBls12377.FreeAsync(streamBls12377)

	runtime.SynchronizeStream(streamBls12377)

	println(time.Since(startTime).String())
}
