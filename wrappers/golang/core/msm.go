package core

import (
    "fmt"
	"local/hello/icicle/wrappers/golang/cuda_runtime"
)

type MSMConfig struct {
    /// Details related to the device such as its id and stream.
    Ctx cuda_runtime.DeviceContext

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

    /// Whether to run the MSM asyncronously. If set to `true`, the MSM function will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, the MSM function will block the current CPU thread.
    IsAsync bool
}

// type MSM interface {
// 	Msm(scalars, points *cuda_runtime.HostOrDeviceSlice, cfg *MSMConfig, results *cuda_runtime.HostOrDeviceSlice) cuda_runtime.CudaError
// 	GetDefaultMSMConfig() MSMConfig
// }

func GetDefaultMSMConfig() MSMConfig {
    ctx, _ := cuda_runtime.GetDefaultDeviceContext()
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

func MsmCheck(scalars cuda_runtime.HostOrDeviceSlice[any, any], points cuda_runtime.HostOrDeviceSlice[any, any], cfg *MSMConfig, results cuda_runtime.HostOrDeviceSlice[any, any]) {
    scalarsLength, pointsLength, resultsLength := scalars.Len(), points.Len(), results.Len()
    if scalarsLength % pointsLength != 0 {
        errorString := fmt.Sprintf(
            "Number of points %d does not divide the number of scalars %d",
            pointsLength,
            scalarsLength,
        )
        panic(errorString)
    }
    if scalarsLength % resultsLength != 0 {
        errorString := fmt.Sprintf(
            "Number of results %d does not divide the number of scalars %d",
            resultsLength,
            scalarsLength,
        )
        panic(errorString)
    }
    cfg.pointsSize = int32(pointsLength)
    cfg.batchSize = int32(resultsLength)
    cfg.areScalarsOnDevice = scalars.IsOnDevice();
    cfg.arePointsOnDevice = points.IsOnDevice();
    cfg.areResultsOnDevice = results.IsOnDevice();
}
