package core

import (
	"fmt"
	"unsafe"

	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

type MSMConfig struct {
	/// Details related to the device such as its id and stream.
	Ctx cr.DeviceContext

	pointsSize int32

	/// The number of extra points to pre-compute for each point. Larger values decrease the number of computations
	/// to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute).
	PrecomputeFactor int32

	/// `c` value, or "window bitsize" which is the main parameter of the "bucket method"
	/// that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
	/// footprint but also more parallelism and less computational complexity (up to a certain point).
	/// Default value: 0 (the optimal value of `c` is chosen automatically).
	C int32

	/// Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different
	/// (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field).
	Bitsize int32

	/// Variable that controls how sensitive the algorithm is to the buckets that occur very frequently.
	/// Useful for efficient treatment of non-uniform distributions of scalars and "top windows" with few bits.
	/// Can be set to 0 to disable separate treatment of large buckets altogether. Default value: 10.
	LargeBucketFactor int32

	batchSize int32

	areScalarsOnDevice bool

	/// True if scalars are in Montgomery form and false otherwise. Default value: true.
	AreScalarsMontgomeryForm bool

	arePointsOnDevice bool

	/// True if coordinates of points are in Montgomery form and false otherwise. Default value: true.
	ArePointsMontgomeryForm bool

	areResultsOnDevice bool

	/// Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly
	/// decreases parallelism, so only suitable for large batches of MSMs. Default value: false.
	IsBigTriangle bool

	/// Whether to run the MSM asynchronously. If set to `true`, the MSM function will be non-blocking
	/// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
	/// If set to `false`, the MSM function will block the current CPU thread.
	IsAsync bool
}

func GetDefaultMSMConfig() MSMConfig {
	ctx, _ := cr.GetDefaultDeviceContext()
	return MSMConfig{
		ctx,   // Ctx
		0,     // pointsSize
		1,     // PrecomputeFactor
		0,     // C
		0,     // Bitsize
		10,    // LargeBucketFactor
		1,     // batchSize
		false, // areScalarsOnDevice
		false, // AreScalarsMontgomeryForm
		false, // arePointsOnDevice
		false, // ArePointsMontgomeryForm
		false, // areResultsOnDevice
		false, // IsBigTriangle
		false, // IsAsync
	}
}

func MsmCheck(scalars HostOrDeviceSlice, points HostOrDeviceSlice, cfg *MSMConfig, results HostOrDeviceSlice) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int, unsafe.Pointer) {
	scalarsLength, pointsLength, resultsLength := scalars.Len(), points.Len()/int(cfg.PrecomputeFactor), results.Len()
	if scalarsLength%pointsLength != 0 {
		errorString := fmt.Sprintf(
			"Number of points %d does not divide the number of scalars %d",
			pointsLength,
			scalarsLength,
		)
		panic(errorString)
	}
	if scalarsLength%resultsLength != 0 {
		errorString := fmt.Sprintf(
			"Number of results %d does not divide the number of scalars %d",
			resultsLength,
			scalarsLength,
		)
		panic(errorString)
	}
	cfg.pointsSize = int32(pointsLength)
	cfg.batchSize = int32(resultsLength)
	cfg.areScalarsOnDevice = scalars.IsOnDevice()
	cfg.arePointsOnDevice = points.IsOnDevice()
	cfg.areResultsOnDevice = results.IsOnDevice()

	if scalars.IsOnDevice() {
		scalars.(DeviceSlice).CheckDevice()
	}

	if points.IsOnDevice() {
		points.(DeviceSlice).CheckDevice()
	}

	if results.IsOnDevice() {
		results.(DeviceSlice).CheckDevice()
	}

	size := scalars.Len() / results.Len()
	return scalars.AsUnsafePointer(), points.AsUnsafePointer(), results.AsUnsafePointer(), size, unsafe.Pointer(cfg)
}

func PrecomputeBasesCheck(points HostOrDeviceSlice, precomputeFactor int32, outputBases DeviceSlice) (unsafe.Pointer, unsafe.Pointer) {
	outputBasesLength, pointsLength := outputBases.Len(), points.Len()
	if outputBasesLength != pointsLength*int(precomputeFactor) {
		errorString := fmt.Sprintf(
			"Precompute factor is probably incorrect: expected %d but got %d",
			outputBasesLength/pointsLength,
			precomputeFactor,
		)
		panic(errorString)
	}

	if points.IsOnDevice() {
		points.(DeviceSlice).CheckDevice()
	}

	return points.AsUnsafePointer(), outputBases.AsUnsafePointer()
}
