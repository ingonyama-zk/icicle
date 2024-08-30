package core

import (
	"fmt"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
)

const CUDA_MSM_LARGE_BUCKET_FACTOR = "large_bucket_factor"
const CUDA_MSM_IS_BIG_TRIANGLE = "is_big_triangle"

type MSMConfig struct {
	/// Specifies the stream (queue) to use for async execution.
	StreamHandle runtime.Stream
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

	BatchSize int32

	/// Bases are shared for batch. Set to true if all MSMs use the same bases. Otherwise, the number
	/// of bases and number of scalars are expected to be equal. Default value: true.
	ArePointsSharedInBatch   bool
	areScalarsOnDevice       bool
	AreScalarsMontgomeryForm bool
	areBasesOnDevice         bool
	AreBasesMontgomeryForm   bool
	areResultsOnDevice       bool

	/// Whether to run the MSM asynchronously. If set to `true`, the MSM function will be non-blocking
	/// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
	/// If set to `false`, the MSM function will block the current CPU thread.
	IsAsync bool
	Ext     config_extension.ConfigExtensionHandler
}

func GetDefaultMSMConfig() MSMConfig {
	return MSMConfig{
		StreamHandle:             nil,
		PrecomputeFactor:         1,
		C:                        0,
		Bitsize:                  0,
		BatchSize:                1,
		ArePointsSharedInBatch:   true,
		areScalarsOnDevice:       false,
		AreScalarsMontgomeryForm: false,
		areBasesOnDevice:         false,
		AreBasesMontgomeryForm:   false,
		areResultsOnDevice:       false,
		IsAsync:                  false,
		Ext:                      nil,
	}
}

func MsmCheck(scalars HostOrDeviceSlice, bases HostOrDeviceSlice, cfg *MSMConfig, results HostOrDeviceSlice) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int) {
	if bases.Len()%int(cfg.PrecomputeFactor) != 0 {
		errorString := fmt.Sprintf(
			"Precompute factor %d does not divide the number of bases %d",
			cfg.PrecomputeFactor,
			bases.Len(),
		)
		panic(errorString)
	}
	scalarsLength, basesLength, resultsLength := scalars.Len(), bases.Len()/int(cfg.PrecomputeFactor), results.Len()
	if scalarsLength%basesLength != 0 {
		errorString := fmt.Sprintf(
			"Number of bases %d does not divide the number of scalars %d",
			basesLength,
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

	// cfg.basesSize = int32(basesLength)
	cfg.ArePointsSharedInBatch = basesLength < scalarsLength
	cfg.BatchSize = int32(resultsLength)
	cfg.areScalarsOnDevice = scalars.IsOnDevice()
	cfg.areBasesOnDevice = bases.IsOnDevice()
	cfg.areResultsOnDevice = results.IsOnDevice()

	if scalars.IsOnDevice() {
		scalars.(DeviceSlice).CheckDevice()
	}

	if bases.IsOnDevice() {
		bases.(DeviceSlice).CheckDevice()
	}

	if results.IsOnDevice() {
		results.(DeviceSlice).CheckDevice()
	}

	size := scalars.Len() / results.Len()
	return scalars.AsUnsafePointer(), bases.AsUnsafePointer(), results.AsUnsafePointer(), size
}

func PrecomputeBasesCheck(bases HostOrDeviceSlice, cfg *MSMConfig, outputBases DeviceSlice) (unsafe.Pointer, unsafe.Pointer) {
	outputBasesLength, basesLength := outputBases.Len(), bases.Len()
	if outputBasesLength != basesLength*int(cfg.PrecomputeFactor) {
		errorString := fmt.Sprintf(
			"Precompute factor is probably incorrect: expected %d but got %d",
			outputBasesLength/basesLength,
			cfg.PrecomputeFactor,
		)
		panic(errorString)
	}

	if bases.IsOnDevice() {
		bases.(DeviceSlice).CheckDevice()
	}
	outputBases.CheckDevice()

	cfg.areBasesOnDevice = bases.IsOnDevice()
	cfg.areResultsOnDevice = true

	return bases.AsUnsafePointer(), outputBases.AsUnsafePointer()
}
