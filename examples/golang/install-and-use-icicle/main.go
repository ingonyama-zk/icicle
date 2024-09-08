package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"
	bls12377G2 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/g2"
	bls12377Msm "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/msm"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	bn254G2 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/g2"
	bn254Msm "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/msm"
)

func main() {
	runtime.LoadBackendFromEnvOrDefault()

	var logSizeMin int
	var logSizeMax int
	var deviceType string

	flag.IntVar(&logSizeMin, "l", 10, "Minimum log size")
	flag.IntVar(&logSizeMax, "u", 10, "Maximum log size")
	flag.StringVar(&deviceType, "device", "CUDA", "Device type")
	flag.Parse()

	device := runtime.CreateDevice(deviceType, 0)
	runtime.SetDevice(&device)

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

		streamBn254, _ := runtime.CreateStream()
		streamBn254G2, _ := runtime.CreateStream()

		cfgBn254.StreamHandle = streamBn254
		cfgBn254G2.StreamHandle = streamBn254G2

		var projectiveBn254 bn254.Projective
		var projectiveBn254G2 bn254G2.G2Projective

		var msmResultBn254 core.DeviceSlice
		var msmResultBn254G2 core.DeviceSlice

		_, e := msmResultBn254.MallocAsync(projectiveBn254.Size(), 1, streamBn254)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"Bn254 Malloc failed: ", e)
			panic(errorString)
		}
		_, e = msmResultBn254G2.MallocAsync(projectiveBn254G2.Size(), 1, streamBn254G2)
		if e != runtime.Success {
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

		streamBls12377, _ := runtime.CreateStream()
		streamBls12377G2, _ := runtime.CreateStream()

		cfgBls12377.StreamHandle = streamBls12377
		cfgBls12377G2.StreamHandle = streamBls12377G2

		var projectiveBls12377 bls12377.Projective
		var projectiveBls12377G2 bls12377G2.G2Projective

		var msmResultBls12377 core.DeviceSlice
		var msmResultBls12377G2 core.DeviceSlice

		_, e = msmResultBls12377.MallocAsync(projectiveBls12377.Size(), 1, streamBls12377)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"Bls12_377 Malloc failed: ", e)
			panic(errorString)
		}
		_, e = msmResultBls12377G2.MallocAsync(projectiveBls12377G2.Size(), 1, streamBls12377G2)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"Bls12_377 Malloc G2 failed: ", e)
			panic(errorString)
		}

		println(time.Since(startTime).String())

		print("Executing bn254 MSM on device ... ")
		startTime = time.Now()

		currentDevice, _ := runtime.GetActiveDevice()
		print("Device: ", currentDevice.GetDeviceType())

		e = bn254Msm.Msm(scalarsBn254, pointsBn254, &cfgBn254, msmResultBn254)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"bn254 Msm failed: ", e)
			panic(errorString)
		}
		e = bn254G2.G2Msm(scalarsBn254, pointsBn254G2, &cfgBn254G2, msmResultBn254G2)
		if e != runtime.Success {
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

		runtime.SynchronizeStream(streamBn254)
		runtime.SynchronizeStream(streamBn254G2)

		println(time.Since(startTime).String())

		print("Executing Bls12377 MSM on device ... ")
		startTime = time.Now()

		currentDevice, _ = runtime.GetActiveDevice()
		print("Device: ", currentDevice.GetDeviceType())

		e = bls12377Msm.Msm(scalarsBls12377, pointsBls12377, &cfgBls12377, msmResultBls12377)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"bls12_377 Msm failed: ", e)
			panic(errorString)
		}
		e = bls12377G2.G2Msm(scalarsBls12377, pointsBls12377G2, &cfgBls12377G2, msmResultBls12377G2)
		if e != runtime.Success {
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

		runtime.SynchronizeStream(streamBls12377)
		runtime.SynchronizeStream(streamBls12377G2)

		println(time.Since(startTime).String())
	}
}
