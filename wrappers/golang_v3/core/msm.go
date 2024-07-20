package core

import (
	"fmt"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
)

type MSMConfig struct {
	StreamHandle runtime.Stream
	basesSize    int32

	/// The number of extra bases to pre-compute for each point. See the `precompute_bases` function, `precompute_factor` passed
	/// there needs to be equal to the one used here. Larger values decrease the number of computations
	/// to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute).
	///
	PrecomputeFactor int32

	/// `c` value, or "window bitsize" which is the main parameter of the "bucket method"
	/// that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
	/// footprint but also more parallelism and less computational complexity (up to a certain point).
	/// Currently pre-computation is independent of `c`, however in the future value of `c` here and the one passed into the
	/// `precompute_bases` function will need to be identical. Default value: 0 (the optimal value of `c` is chosen automatically).
	C int32

	/// Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different
	/// (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field).
	Bitsize int32

	batchSize                int32
	areScalarsOnDevice       bool
	AreScalarsMontgomeryForm bool
	areBasesOnDevice         bool
	AreBasesMontgomeryForm   bool
	areResultsOnDevice       bool

	/// Whether to run the MSM asynchronously. If set to `true`, the MSM function will be non-blocking
	/// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
	/// If set to `false`, the MSM function will block the current CPU thread.
	IsAsync bool
	Ext     runtime.ConfigExtensionHandler
}

func GetDefaultMSMConfig() MSMConfig {
	return MSMConfig{
		nil,   // StreamHandle
		0,     // basesSize
		1,     // PrecomputeFactor
		0,     // C
		0,     // Bitsize
		1,     // 	batchSize
		false, // areScalarsOnDevice
		false, // AreScalarsMontgomeryForm
		false, // areBasesOnDevice
		false, // AreBasesMontgomeryForm
		false, // areResultsOnDevice
		false, // 	IsAsync
		nil,   // Ext
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

	cfg.areScalarsOnDevice = scalars.IsOnDevice()
	cfg.areBasesOnDevice = points.IsOnDevice()
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

func PrecomputePointsCheck(points HostOrDeviceSlice, cfg *MSMConfig, outputBases DeviceSlice) (unsafe.Pointer, unsafe.Pointer) {
	outputBasesLength, pointsLength := outputBases.Len(), points.Len()
	if outputBasesLength != pointsLength*int(cfg.PrecomputeFactor) {
		errorString := fmt.Sprintf(
			"Precompute factor is probably incorrect: expected %d but got %d",
			outputBasesLength/pointsLength,
			cfg.PrecomputeFactor,
		)
		panic(errorString)
	}

	if points.IsOnDevice() {
		points.(DeviceSlice).CheckDevice()
	}

	cfg.basesSize = int32(pointsLength)
	cfg.areBasesOnDevice = points.IsOnDevice()

	return points.AsUnsafePointer(), outputBases.AsUnsafePointer()
}
