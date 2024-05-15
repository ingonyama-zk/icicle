package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"

	bls12377Ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/ntt"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254"

	bn254Ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254/ntt"

	bls12377Fft "github.com/consensys/gnark-crypto/ecc/bls12-377/fr/fft"
	bn254Fft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func main() {
	var logSizeMin int
	var logSizeMax int

	flag.IntVar(&logSizeMin, "logSizeMin", 17, "")
	flag.IntVar(&logSizeMax, "logSizeMax", 22, "")
	flag.Parse()

	size := 1 << logSizeMax

	print("Generating BN254 scalars ... ")
	startTime := time.Now()
	scalarsBn254Max := bn254.GenerateScalars(size)
	println(time.Since(startTime).String())

	cfgBn254 := bn254Ntt.GetDefaultNttConfig()
	cfgBn254.IsAsync = true

	print("Generating BLS12_377 scalars ... ")
	startTime = time.Now()
	scalarsBls12377Max := bls12377.GenerateScalars(size)
	println(time.Since(startTime).String())

	cfgBls12377 := bls12377Ntt.GetDefaultNttConfig()
	cfgBls12377.IsAsync = true

	rouMontBn254, _ := bn254Fft.Generator(uint64(size))
	rouBn254 := rouMontBn254.Bits()
	rouIcicleBn254 := bn254.ScalarField{}
	limbsBn254 := core.ConvertUint64ArrToUint32Arr(rouBn254[:])
	rouIcicleBn254.FromLimbs(limbsBn254)
	bn254Ntt.InitDomain(rouIcicleBn254, cfgBn254.Ctx, false)

	rouMontBls12377, _ := bls12377Fft.Generator(uint64(size))
	rouBls12377 := rouMontBls12377.Bits()
	rouIcicleBls12377 := bls12377.ScalarField{}
	limbsBls12377 := core.ConvertUint64ArrToUint32Arr(rouBls12377[:])
	rouIcicleBls12377.FromLimbs(limbsBls12377)
	bls12377Ntt.InitDomain(rouIcicleBls12377, cfgBls12377.Ctx, false)

	for logSize := logSizeMin; logSize < logSizeMax; logSize++ {

		// Define the size of the problem, here 2^18.
		size := 1 << logSize

		fmt.Printf("---------------------- NTT size 2^%d=%d ------------------------\n", logSize, size)

		print("Configuring bn254 NTT ... ")
		startTime = time.Now()

		scalarsBn254 := scalarsBn254Max[:size]

		streamBn254, _ := cr.CreateStream()

		cfgBn254.Ctx.Stream = &streamBn254

		var nttResultBn254 core.DeviceSlice

		_, e := nttResultBn254.MallocAsync(size*scalarsBn254.SizeOfElement(), scalarsBn254.SizeOfElement(), streamBn254)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"Bn254 Malloc failed: ", e)
			panic(errorString)
		}

		println(time.Since(startTime).String())

		print("Configuring Bls12377 NTT ... ")
		startTime = time.Now()

		scalarsBls12377 := scalarsBls12377Max[:size]

		streamBls12377, _ := cr.CreateStream()

		cfgBls12377.Ctx.Stream = &streamBls12377

		var msmResultBls12377 core.DeviceSlice

		_, e = msmResultBls12377.MallocAsync(size*scalarsBls12377.SizeOfElement(), scalarsBls12377.SizeOfElement(), streamBls12377)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"Bls12_377 Malloc failed: ", e)
			panic(errorString)
		}

		println(time.Since(startTime).String())

		print("Executing bn254 NTT on device ... ")
		startTime = time.Now()

		err := bn254Ntt.Ntt(scalarsBn254, core.KForward, &cfgBn254, nttResultBn254)
		if err.CudaErrorCode != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"bn254 Ntt failed: ", e)
			panic(errorString)
		}

		nttResultBn254Host := make(core.HostSlice[bn254.ScalarField], size)
		nttResultBn254Host.CopyFromDeviceAsync(&nttResultBn254, streamBn254)
		nttResultBn254.FreeAsync(streamBn254)
		cr.SynchronizeStream(&streamBn254)
		println(time.Since(startTime).String())

		print("Executing Bls12377 NTT on device ... ")
		startTime = time.Now()

		err = bls12377Ntt.Ntt(scalarsBls12377, core.KForward, &cfgBls12377, msmResultBls12377)
		if err.CudaErrorCode != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"bls12_377 Ntt failed: ", e)
			panic(errorString)
		}

		msmResultBls12377Host := make(core.HostSlice[bls12377.ScalarField], size)

		msmResultBls12377Host.CopyFromDeviceAsync(&msmResultBls12377, streamBls12377)

		msmResultBls12377.FreeAsync(streamBls12377)

		cr.SynchronizeStream(&streamBls12377)

		println(time.Since(startTime).String())
	}
}
