package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"

	bls12377G2 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/g2"
	bls12377Msm "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/msm"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254"

	bn254G2 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254/g2"
	bn254Msm "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254/msm"
)

func main() {
	var logSizeMin int
	var logSizeMax int

	flag.IntVar(&logSizeMin, "l", 17, "Minimum log size")
	flag.IntVar(&logSizeMax, "u", 22, "Maximum log size")
	flag.Parse()

	sizeMax := 1 << logSizeMax

	print("Generating BN254 scalars ... ")
	startTime := time.Now()
	scalarsBn254Max := bn254.GenerateScalars(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BN254 points ... ")
	startTime = time.Now()
	pointsBn254Max := bn254.GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BN254 G2 points ... ")
	startTime = time.Now()
	pointsBn254G2Max := bn254G2.G2GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BLS12_377 scalars ... ")
	startTime = time.Now()
	scalarsBls12377Max := bls12377.GenerateScalars(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BLS12_377 points ... ")
	startTime = time.Now()
	pointsBls12377Max := bls12377.GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BLS12_377 G2 points ... ")
	startTime = time.Now()
	pointsBls12377G2Max := bls12377G2.G2GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	for logSize := logSizeMin; logSize <= logSizeMax; logSize++ {

		// Define the size of the problem, here 2^18.
		size := 1 << logSize

		fmt.Printf("---------------------- MSM size 2^%d=%d ------------------------\n", logSize, size)

		// println(scalarsBls12377, pointsBls12377, pointsBn254G2)
		// println(scalarsBn254, pointsBn254, pointsBls12377G2)

		print("Configuring bn254 MSM ... ")
		startTime = time.Now()

		scalarsBn254 := scalarsBn254Max[:size]
		pointsBn254 := pointsBn254Max[:size]
		pointsBn254G2 := pointsBn254G2Max[:size]

		cfgBn254 := core.GetDefaultMSMConfig()
		cfgBn254G2 := core.GetDefaultMSMConfig()
		cfgBn254.IsAsync = true
		cfgBn254G2.IsAsync = true

		streamBn254, _ := cr.CreateStream()
		streamBn254G2, _ := cr.CreateStream()

		cfgBn254.Ctx.Stream = &streamBn254
		cfgBn254G2.Ctx.Stream = &streamBn254G2

		var projectiveBn254 bn254.Projective
		var projectiveBn254G2 bn254G2.G2Projective

		var msmResultBn254 core.DeviceSlice
		var msmResultBn254G2 core.DeviceSlice

		_, e := msmResultBn254.MallocAsync(projectiveBn254.Size(), projectiveBn254.Size(), streamBn254)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"Bn254 Malloc failed: ", e)
			panic(errorString)
		}
		_, e = msmResultBn254G2.MallocAsync(projectiveBn254G2.Size(), projectiveBn254G2.Size(), streamBn254G2)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"Bn254 Malloc G2 failed: ", e)
			panic(errorString)
		}

		println(time.Since(startTime).String())

		print("Configuring Bls12377 MSM ... ")
		startTime = time.Now()

		scalarsBls12377 := scalarsBls12377Max[:size]
		pointsBls12377 := pointsBls12377Max[:size]
		pointsBls12377G2 := pointsBls12377G2Max[:size]

		cfgBls12377 := core.GetDefaultMSMConfig()
		cfgBls12377G2 := core.GetDefaultMSMConfig()
		cfgBls12377.IsAsync = true
		cfgBls12377G2.IsAsync = true

		streamBls12377, _ := cr.CreateStream()
		streamBls12377G2, _ := cr.CreateStream()

		cfgBls12377.Ctx.Stream = &streamBls12377
		cfgBls12377G2.Ctx.Stream = &streamBls12377G2

		var projectiveBls12377 bls12377.Projective
		var projectiveBls12377G2 bls12377G2.G2Projective

		var msmResultBls12377 core.DeviceSlice
		var msmResultBls12377G2 core.DeviceSlice

		_, e = msmResultBls12377.MallocAsync(projectiveBls12377.Size(), projectiveBls12377.Size(), streamBls12377)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"Bls12_377 Malloc failed: ", e)
			panic(errorString)
		}
		_, e = msmResultBls12377G2.MallocAsync(projectiveBls12377G2.Size(), projectiveBls12377G2.Size(), streamBls12377G2)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"Bls12_377 Malloc G2 failed: ", e)
			panic(errorString)
		}

		println(time.Since(startTime).String())

		print("Executing bn254 MSM on device ... ")
		startTime = time.Now()

		e = bn254Msm.Msm(scalarsBn254, pointsBn254, &cfgBn254, msmResultBn254)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"bn254 Msm failed: ", e)
			panic(errorString)
		}
		e = bn254G2.G2Msm(scalarsBn254, pointsBn254G2, &cfgBn254G2, msmResultBn254G2)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"bn254 Msm G2 failed: ", e)
			panic(errorString)
		}

		msmResultBn254Host := make(core.HostSlice[bn254.Projective], 1)
		msmResultBn254G2Host := make(core.HostSlice[bn254G2.G2Projective], 1)

		msmResultBn254Host.CopyFromDeviceAsync(&msmResultBn254, streamBn254)
		msmResultBn254G2Host.CopyFromDeviceAsync(&msmResultBn254G2, streamBn254G2)

		msmResultBn254.FreeAsync(streamBn254)
		msmResultBn254G2.FreeAsync(streamBn254G2)

		cr.SynchronizeStream(&streamBn254)
		cr.SynchronizeStream(&streamBn254G2)

		println(time.Since(startTime).String())

		print("Executing Bls12377 MSM on device ... ")
		startTime = time.Now()

		e = bls12377Msm.Msm(scalarsBls12377, pointsBls12377, &cfgBls12377, msmResultBls12377)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"bls12_377 Msm failed: ", e)
			panic(errorString)
		}
		e = bls12377G2.G2Msm(scalarsBls12377, pointsBls12377G2, &cfgBls12377G2, msmResultBls12377G2)
		if e != cr.CudaSuccess {
			errorString := fmt.Sprint(
				"bls12_377 Msm G2 failed: ", e)
			panic(errorString)
		}

		msmResultBls12377Host := make(core.HostSlice[bls12377.Projective], 1)
		msmResultBls12377G2Host := make(core.HostSlice[bls12377G2.G2Projective], 1)

		msmResultBls12377Host.CopyFromDeviceAsync(&msmResultBls12377, streamBls12377)
		msmResultBls12377G2Host.CopyFromDeviceAsync(&msmResultBls12377G2, streamBls12377G2)

		msmResultBls12377.FreeAsync(streamBls12377)
		msmResultBls12377G2.FreeAsync(streamBls12377G2)

		cr.SynchronizeStream(&streamBls12377)
		cr.SynchronizeStream(&streamBls12377G2)

		println(time.Since(startTime).String())
	}
}
