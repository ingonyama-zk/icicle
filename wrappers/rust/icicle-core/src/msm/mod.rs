use icicle_cuda_runtime::{device_context::DeviceContext, error::CudaResult};

use crate::curve::{Affine, CurveConfig, Projective};

pub mod tests;

/*
/**
 * @struct MSMConfig
 * Struct that encodes MSM parameters to be passed into the [msm](@ref msm) function.
 */
struct MSMConfig {
  bool are_scalars_on_device;         /**< True if scalars are on device and false if they're on host. Default value: false. */
  bool are_scalars_montgomery_form;   /**< True if scalars are in Montgomery form and false otherwise. Default value: true. */
  int points_size;                    /**< Number of points in the MSM. If a batch of MSMs needs to be computed, this should be a number
                                       *   of different points. So, if each MSM re-uses the same set of points, this variable is set equal
                                       *   to the MSM size. And if every MSM uses a distinct set of points, it should be set to the product of
                                       *   MSM size and [batch_size](@ref batch_size). Default value: 0 (meaning it's equal to the MSM size). */
  int precompute_factor;              /**< The number of extra points to pre-compute for each point. Larger values decrease the number of computations
                                       *   to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute). */
  bool are_points_on_device;          /**< True if points are on device and false if they're on host. Default value: false. */
  bool are_points_montgomery_form;    /**< True if coordinates of points are in Montgomery form and false otherwise. Default value: true. */
  int batch_size;                     /**< The number of MSMs to compute. Default value: 1. */
  bool are_results_on_device;         /**< True if the results should be on device and false if they should be on host. If set to false,
                                       *   `is_async` won't take effect because a synchronization is needed to transfer results to the host. Default value: false. */
  int c;                              /**< \f$ c \f$ value, or "window bitsize" which is the main parameter of the "bucket method"
                                       *   that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
                                       *   footprint but also more parallelism and less computational complexity (up to a certain point).
                                       *   Default value: 0 (the optimal value of \f$ c \f$ is chosen automatically). */
  int bitsize;                        /**< Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different
                                       *   (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field). */
  bool is_big_triangle;               /**< Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly
                                       *   decreases parallelism, so only suitable for large batches of MSMs. Default value: false. */
  int large_bucket_factor;            /**< Variable that controls how sensitive the algorithm is to the buckets that occur very frequently.
                                       *   Useful for efficient treatment of non-uniform distributions of scalars and "top windows" with few bits.
                                       *   Can be set to 0 to disable separate treatment of large buckets altogether. Default value: 10. */
  int is_async;                       /**< Whether to run the MSM asyncronously. If set to `true`, the MSM function will be non-blocking
                                       *   and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
                                       *   If set to false, the MSM function will block the current CPU thread. */
  device_context::DeviceContext ctx;  /**< Details related to the device such as its id and stream id. See [DeviceContext](@ref `device_context::DeviceContext`). */
};
*/
/// Struct that encodes MSM parameters to be passed into the `msm` function.

#[repr(C)]
pub struct MSMConfig<'a> {
    /// True if scalars are on device and false if they're on host. Default value: false.
    pub are_scalars_on_device: bool,

    /// True if scalars are in Montgomery form and false otherwise. Default value: true.
    pub are_scalars_montgomery_form: bool,

    /// Number of points in the MSM. If a batch of MSMs needs to be computed, this should be a number
    /// of different points. So, if each MSM re-uses the same set of points, this variable is set equal
    /// to the MSM size. And if every MSM uses a distinct set of points, it should be set to the product of
    /// MSM size and batch_size. Default value: 0 (meaning it's equal to the MSM size).
    pub points_size: usize, // Note: `unsigned` in C++ corresponds to `u32` in Rust

    /// The number of extra points to pre-compute for each point. Larger values decrease the number of computations
    /// to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute).
    pub precompute_factor: usize,

    /// True if points are on device and false if they're on host. Default value: false.
    pub are_points_on_device: bool,

    /// True if coordinates of points are in Montgomery form and false otherwise. Default value: true.
    pub are_points_montgomery_form: bool,

    /// The number of MSMs to compute. Default value: 1.
    pub batch_size: usize,

    /// True if the results should be on device and false if they should be on host. If set to false,
    /// `is_async` won't take effect because a synchronization is needed to transfer results to the host. Default value: false.
    pub are_results_on_device: bool,

    /// `c` value, or "window bitsize" which is the main parameter of the "bucket method"
    /// that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
    /// footprint but also more parallelism and less computational complexity (up to a certain point).
    /// Default value: 0 (the optimal value of `c` is chosen automatically).
    pub c: usize,

    /// Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different
    /// (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field).
    pub bitsize: usize,

    /// Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly
    /// decreases parallelism, so only suitable for large batches of MSMs. Default value: false.
    pub is_big_triangle: bool,

    /// Variable that controls how sensitive the algorithm is to the buckets that occur very frequently.
    /// Useful for efficient treatment of non-uniform distributions of scalars and "top windows" with few bits.
    /// Can be set to 0 to disable separate treatment of large buckets altogether. Default value: 10.
    pub large_bucket_factor: usize,

    /// Whether to run the MSM asyncronously. If set to `true`, the MSM function will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, the MSM function will block the current CPU thread.
    pub is_async: bool,

    /// Details related to the device such as its id and stream id.
    pub ctx: DeviceContext<'a>,
}

pub trait MSM<C: CurveConfig> {
    fn msm<'a>(
        scalars: &[C::ScalarField],
        points: &[Affine<C>],
        cfg: MSMConfig<'a>,
        results: &mut [Projective<C>],
    ) -> CudaResult<()>;

    fn get_default_msm_config() -> MSMConfig<'static>;
}

#[macro_export]
macro_rules! impl_msm {
    (
      $curve_prefix:literal,
      $curve_config:ident
    ) => {
        extern "C" {
            #[link_name = concat!($curve_prefix, "MSMCuda")]
            fn msm_cuda<'a>(
                scalars: *const ScalarField,
                points: *const G1Affine,
                count: usize,
                config: MSMConfig<'a>,
                out: *mut G1Projective,
            ) -> CudaError;

            #[link_name = concat!($curve_prefix, "DefaultMSMConfig")]
            fn default_msm_config() -> MSMConfig<'static>;
        }

        impl MSM<$curve_config> for $curve_config {
            fn msm<'a>(
                scalars: &[<$curve_config as CurveConfig>::ScalarField],
                points: &[<$curve_config as CurveConfig>::Affine],
                cfg: MSMConfig<'a>,
                results: &mut [<$curve_config as CurveConfig>::Projective],
            ) -> CudaResult<()> {
                if points.len() != scalars.len() {
                    return Err(CudaError::cudaErrorInvalidValue);
                }

                unsafe {
                    msm_cuda(
                        scalars as *const _ as *const <$curve_config as CurveConfig>::ScalarField,
                        points as *const _ as *const <$curve_config as CurveConfig>::Affine,
                        points.len(),
                        cfg,
                        results as *mut _ as *mut <$curve_config as CurveConfig>::Projective,
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
      $curve_config:ident,
      $scalar_config:ident
    ) => {
        #[test]
        fn test_msm() {
            let log_test_sizes = [20];

            check_msm::<$curve_config, $scalar_config>(&log_test_sizes)
        }
    };
}
