use crate::curve::{Affine, Curve, Projective};
use crate::error::IcicleResult;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;

/// Struct that encodes MSM parameters to be passed into the [`msm`](msm) function.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MSMConfig<'a> {
    /// Details related to the device such as its id and stream.
    pub ctx: DeviceContext<'a>,

    points_size: i32,

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

    batch_size: i32,

    are_scalars_on_device: bool,

    /// True if scalars are in Montgomery form and false otherwise. Default value: true.
    pub are_scalars_montgomery_form: bool,

    are_points_on_device: bool,

    /// True if coordinates of points are in Montgomery form and false otherwise. Default value: true.
    pub are_points_montgomery_form: bool,

    are_results_on_device: bool,

    /// Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly
    /// decreases parallelism, so only suitable for large batches of MSMs. Default value: false.
    pub is_big_triangle: bool,

    /// Whether to run the MSM asynchronously. If set to `true`, the MSM function will be non-blocking
    /// and you'd need to synchronize it explicitly by running `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
    /// If set to `false`, the MSM function will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for MSMConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> MSMConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        Self {
            ctx: DeviceContext::default_for_device(device_id),
            points_size: 0,
            precompute_factor: 1,
            c: 0,
            bitsize: 0,
            large_bucket_factor: 10,
            batch_size: 1,
            are_scalars_on_device: false,
            are_scalars_montgomery_form: false,
            are_points_on_device: false,
            are_points_montgomery_form: false,
            are_results_on_device: false,
            is_big_triangle: false,
            is_async: false,
        }
    }
}

#[doc(hidden)]
pub trait MSM<C: Curve> {
    fn msm_unchecked(
        scalars: &HostOrDeviceSlice<C::ScalarField>,
        points: &HostOrDeviceSlice<Affine<C>>,
        cfg: &MSMConfig,
        results: &mut HostOrDeviceSlice<Projective<C>>,
    ) -> IcicleResult<()>;

    fn precompute_bases_unchecked(
        points: &HostOrDeviceSlice<Affine<C>>,
        precompute_factor: i32,
        ctx: &DeviceContext,
        output_bases: &mut HostOrDeviceSlice<Affine<C>>,
    ) -> IcicleResult<()>;
}

/// Computes the multi-scalar multiplication, or MSM: `s1*P1 + s2*P2 + ... + sn*Pn`, or a batch of several MSMs.
///
/// # Arguments
///
/// * `scalars` - scalar values `s1, s2, ..., sn`.
///
/// * `points` - points `P1, P2, ..., Pn`. The number of points can be smaller than the number of scalars
/// in the case of batch MSM. In this case points are re-used periodically. Alternatively, there can be more points
/// than scalars if precomputation has been performed, you need to set `cfg.precompute_factor` in that case.
///
/// * `cfg` - config used to specify extra arguments of the MSM.
///
/// * `results` - buffer to write results into. Its length is equal to the batch size i.e. number of MSMs to compute.
///
/// Returns `()` if no errors occurred or a `CudaError` otherwise.
pub fn msm<C: Curve + MSM<C>>(
    scalars: &HostOrDeviceSlice<C::ScalarField>,
    points: &HostOrDeviceSlice<Affine<C>>,
    cfg: &MSMConfig,
    results: &mut HostOrDeviceSlice<Projective<C>>,
) -> IcicleResult<()> {
    if points.len() % (cfg.precompute_factor as usize) != 0 {
        panic!(
            "Precompute factor {} does not divide the number of points {}",
            cfg.precompute_factor,
            points.len()
        );
    }
    let points_size = points.len() / (cfg.precompute_factor as usize);
    if scalars.len() % points_size != 0 {
        panic!(
            "Number of points {} does not divide the number of scalars {}",
            points_size,
            scalars.len()
        );
    }
    if scalars.len() % results.len() != 0 {
        panic!(
            "Number of results {} does not divide the number of scalars {}",
            results.len(),
            scalars.len()
        );
    }
    let mut local_cfg = cfg.clone();
    local_cfg.points_size = points_size as i32;
    local_cfg.batch_size = results.len() as i32;
    local_cfg.are_scalars_on_device = scalars.is_on_device();
    local_cfg.are_points_on_device = points.is_on_device();
    local_cfg.are_results_on_device = results.is_on_device();

    C::msm_unchecked(scalars, points, &local_cfg, results)
}

/// A function that precomputes MSM bases by extending them with their shifted copies.
/// e.g.:
/// Original points: \f$ P_0, P_1, P_2, ... P_{size} \f$
/// Extended points: \f$ P_0, P_1, P_2, ... P_{size}, 2^{l}P_0, 2^{l}P_1, ..., 2^{l}P_{size},
/// 2^{2l}P_0, 2^{2l}P_1, ..., 2^{2cl}P_{size}, ... \f$
///
/// * `bases` - Bases \f$ P_i \f$. In case of batch MSM, all *unique* points are concatenated.
///
/// * `precompute_factor` - The number of total shifted sets of bases (including the original one).
///
/// * `ctx` - Device context specifying device id and stream to use.
///
/// * `output_bases` - Device-allocated buffer of size bases_size * precompute_factor for the extended bases.
///
/// Returns `()` if no errors occurred or a `CudaError` otherwise.
pub fn precompute_bases<C: Curve + MSM<C>>(
    points: &HostOrDeviceSlice<Affine<C>>,
    precompute_factor: i32,
    ctx: &DeviceContext,
    output_bases: &mut HostOrDeviceSlice<Affine<C>>,
) -> IcicleResult<()> {
    assert_eq!(
        output_bases.len(),
        points.len() * (precompute_factor as usize),
        "Precompute factor is probably incorrect: expected {} but got {}",
        output_bases.len() / points.len(),
        precompute_factor
    );
    assert!(output_bases.is_on_device());

    C::precompute_bases_unchecked(points, precompute_factor, ctx, output_bases)
}

#[macro_export]
macro_rules! impl_msm {
    (
      $curve_prefix:literal,
      $curve_prefix_indent:ident,
      $curve:ident
    ) => {
        mod $curve_prefix_indent {
            use super::{$curve, Affine, CudaError, Curve, DeviceContext, MSMConfig, Projective};

            extern "C" {
                #[link_name = concat!($curve_prefix, "MSMCuda")]
                pub(crate) fn msm_cuda(
                    scalars: *const <$curve as Curve>::ScalarField,
                    points: *const Affine<$curve>,
                    count: i32,
                    config: &MSMConfig,
                    out: *mut Projective<$curve>,
                ) -> CudaError;

                #[link_name = concat!($curve_prefix, "PrecomputeMSMBases")]
                pub(crate) fn precompute_bases_cuda(
                    points: *const Affine<$curve>,
                    bases_size: i32,
                    precompute_factor: i32,
                    are_bases_on_device: bool,
                    ctx: &DeviceContext,
                    output_bases: *mut Affine<$curve>,
                ) -> CudaError;
            }
        }

        impl MSM<$curve> for $curve {
            fn msm_unchecked(
                scalars: &HostOrDeviceSlice<<$curve as Curve>::ScalarField>,
                points: &HostOrDeviceSlice<Affine<$curve>>,
                cfg: &MSMConfig,
                results: &mut HostOrDeviceSlice<Projective<$curve>>,
            ) -> IcicleResult<()> {
                unsafe {
                    $curve_prefix_indent::msm_cuda(
                        scalars.as_ptr(),
                        points.as_ptr(),
                        (scalars.len() / results.len()) as i32,
                        cfg,
                        results.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn precompute_bases_unchecked(
                points: &HostOrDeviceSlice<Affine<$curve>>,
                precompute_factor: i32,
                ctx: &DeviceContext,
                output_bases: &mut HostOrDeviceSlice<Affine<$curve>>,
            ) -> IcicleResult<()> {
                unsafe {
                    $curve_prefix_indent::precompute_bases_cuda(
                        points.as_ptr(),
                        points.len() as i32,
                        precompute_factor,
                        points.is_on_device(),
                        ctx,
                        output_bases.as_mut_ptr(),
                    )
                    .wrap()
                }
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
