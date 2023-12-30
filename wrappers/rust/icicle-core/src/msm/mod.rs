use crate::curve::{Affine, Curve, Projective};
use crate::error::IcicleResult;
use icicle_cuda_runtime::device_context::DeviceContext;

#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;

/// Struct that encodes MSM parameters to be passed into the `msm` function.
#[repr(C)]
#[derive(Debug)]
pub struct MSMConfig<'a> {
    /// Details related to the device such as its id and stream id.
    pub ctx: DeviceContext<'a>,

    /// Number of points in the MSM. If a batch of MSMs needs to be computed, this should be a number
    /// of different points. So, if each MSM re-uses the same set of points, this variable is set equal
    /// to the MSM size. And if every MSM uses a distinct set of points, it should be set to the product of
    /// MSM size and batch_size. Default value: 0 (meaning it's equal to the MSM size).
    pub points_size: i32,

    /// The number of extra points to pre-compute for each point. Larger values decrease the number of computations
    /// to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute).
    pub precompute_factor: i32,

    /// `c` value, or "window bitsize" which is the main parameter of the "bucket method"
    /// that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
    /// footprint but also more parallelism and less computational complexity (up to a certain point).
    /// Default value: 0 (the optimal value of `c` is chosen automatically).
    pub c: i32,

    /// Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different
    /// (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field).
    pub bitsize: i32,

    /// Variable that controls how sensitive the algorithm is to the buckets that occur very frequently.
    /// Useful for efficient treatment of non-uniform distributions of scalars and "top windows" with few bits.
    /// Can be set to 0 to disable separate treatment of large buckets altogether. Default value: 10.
    pub large_bucket_factor: i32,

    /// The number of MSMs to compute. Default value: 1.
    pub batch_size: i32,

    /// True if scalars are on device and false if they're on host. Default value: false.
    pub are_scalars_on_device: bool,

    /// True if scalars are in Montgomery form and false otherwise. Default value: true.
    pub are_scalars_montgomery_form: bool,

    /// True if points are on device and false if they're on host. Default value: false.
    pub are_points_on_device: bool,

    /// True if coordinates of points are in Montgomery form and false otherwise. Default value: true.
    pub are_points_montgomery_form: bool,

    /// True if the results should be on device and false if they should be on host. If set to false,
    /// `is_async` won't take effect because a synchronization is needed to transfer results to the host. Default value: false.
    pub are_results_on_device: bool,

    /// Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly
    /// decreases parallelism, so only suitable for large batches of MSMs. Default value: false.
    pub is_big_triangle: bool,

    /// Whether to run the MSM asyncronously. If set to `true`, the MSM function will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, the MSM function will block the current CPU thread.
    pub is_async: bool,
}

#[doc(hidden)]
pub trait MSM<C: Curve> {
    fn msm(
        scalars: &[C::ScalarField],
        points: &[Affine<C>],
        cfg: &MSMConfig,
        results: &mut [Projective<C>],
    ) -> IcicleResult<()>;

    fn get_default_msm_config() -> MSMConfig<'static>;
}

pub fn msm<C: Curve + MSM<C>>(
    scalars: &[C::ScalarField],
    points: &[Affine<C>],
    cfg: &MSMConfig,
    results: &mut [Projective<C>],
) -> IcicleResult<()> {
    C::msm(scalars, points, cfg, results)
}

pub fn get_default_msm_config<C: Curve + MSM<C>>() -> MSMConfig<'static> {
    C::get_default_msm_config()
}

#[macro_export]
macro_rules! impl_msm {
    (
      $curve_prefix:literal,
      $curve:ident
    ) => {
        extern "C" {
            #[link_name = concat!($curve_prefix, "MSMCuda")]
            fn msm_cuda(
                scalars: *const <$curve as Curve>::ScalarField,
                points: *const Affine<$curve>,
                count: i32,
                config: &MSMConfig,
                out: *mut Projective<$curve>,
            ) -> CudaError;

            #[link_name = concat!($curve_prefix, "DefaultMSMConfig")]
            fn default_msm_config() -> MSMConfig<'static>;
        }

        impl MSM<$curve> for $curve {
            fn msm(
                scalars: &[<$curve as Curve>::ScalarField],
                points: &[Affine<$curve>],
                cfg: &MSMConfig,
                results: &mut [Projective<$curve>],
            ) -> IcicleResult<()> {
                if (cfg.points_size > 0) && (points.len() != cfg.points_size as usize) {
                    panic!(
                        "Number of points {} and cfg.points_size {} do not match:",
                        points.len(),
                        cfg.points_size
                    );
                }

                unsafe {
                    msm_cuda(
                        scalars as *const _ as *const <$curve as Curve>::ScalarField,
                        points as *const _ as *const Affine<$curve>,
                        (scalars.len() / (cfg.batch_size as usize)) as i32,
                        cfg,
                        results as *mut _ as *mut Projective<$curve>,
                    )
                    .wrap()
                }
            }

            fn get_default_msm_config() -> MSMConfig<'static> {
                unsafe { default_msm_config() }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_msm_tests {
    (
      $curve:ident
    ) => {
        #[test]
        fn test_msm() {
            check_msm::<$curve>()
        }

        #[test]
        fn test_msm_batch() {
            check_msm_batch::<$curve>()
        }

        #[test]
        fn test_msm_skewed_distributions() {
            check_msm_skewed_distributions::<$curve>()
        }
    };
}
