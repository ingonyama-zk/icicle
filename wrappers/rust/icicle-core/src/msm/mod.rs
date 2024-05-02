use crate::curve::{Affine, Curve, Projective};
use crate::error::IcicleResult;
use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::memory::{DeviceSlice, HostOrDeviceSlice};

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

    /// The number of extra points to pre-compute for each point. See the `precompute_bases` function, `precompute_factor` passed
    /// there needs to be equal to the one used here. Larger values decrease the number of computations
    /// to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute).
    pub precompute_factor: i32,

    /// `c` value, or "window bitsize" which is the main parameter of the "bucket method"
    /// that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
    /// footprint but also more parallelism and less computational complexity (up to a certain point).
    /// Currently pre-computation is independent of `c`, however in the future value of `c` here and the one passed into the
    /// `precompute_bases` function will need to be identical. Default value: 0 (the optimal value of `c` is chosen automatically).
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
        scalars: &(impl HostOrDeviceSlice<C::ScalarField> + ?Sized),
        points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
        cfg: &MSMConfig,
        results: &mut (impl HostOrDeviceSlice<Projective<C>> + ?Sized),
    ) -> IcicleResult<()>;

    fn precompute_bases_unchecked(
        points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
        precompute_factor: i32,
        _c: i32,
        ctx: &DeviceContext,
        output_bases: &mut DeviceSlice<Affine<C>>,
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
/// Returns `Ok(())` if no errors occurred or a `CudaError` otherwise.
pub fn msm<C: Curve + MSM<C>>(
    scalars: &(impl HostOrDeviceSlice<C::ScalarField> + ?Sized),
    points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
    cfg: &MSMConfig,
    results: &mut (impl HostOrDeviceSlice<Projective<C>> + ?Sized),
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
    let ctx_device_id = cfg
        .ctx
        .device_id;
    if let Some(device_id) = scalars.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in scalars and context are different"
        );
    }
    if let Some(device_id) = points.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in points and context are different"
        );
    }
    if let Some(device_id) = results.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in results and context are different"
        );
    }
    check_device(ctx_device_id);
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
/// * `precompute_factor` - The number of total precomputed points for each base (including the base itself).
///
/// * `_c` - This is currently unused, but in the future precomputation will need to be aware of
/// the `c` value used in MSM (see MSMConfig). So to avoid breaking your code with this
/// upcoming change, make sure to use the same value of `c` in this function and in respective `MSMConfig`.
///
/// * `ctx` - Device context specifying device id and stream to use.
///
/// * `output_bases` - Device-allocated buffer of size `bases_size` * `precompute_factor` for the extended bases.
///
/// Returns `Ok(())` if no errors occurred or a `CudaError` otherwise.
pub fn precompute_bases<C: Curve + MSM<C>>(
    points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
    precompute_factor: i32,
    _c: i32,
    ctx: &DeviceContext,
    output_bases: &mut DeviceSlice<Affine<C>>,
) -> IcicleResult<()> {
    assert_eq!(
        output_bases.len(),
        points.len() * (precompute_factor as usize),
        "Precompute factor is probably incorrect: expected {} but got {}",
        output_bases.len() / points.len(),
        precompute_factor
    );
    assert!(output_bases.is_on_device());

    C::precompute_bases_unchecked(points, precompute_factor, _c, ctx, output_bases)
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
                #[link_name = concat!($curve_prefix, "_msm_cuda")]
                pub(crate) fn msm_cuda(
                    scalars: *const <$curve as Curve>::ScalarField,
                    points: *const Affine<$curve>,
                    count: i32,
                    config: &MSMConfig,
                    out: *mut Projective<$curve>,
                ) -> CudaError;

                #[link_name = concat!($curve_prefix, "_precompute_msm_bases_cuda")]
                pub(crate) fn precompute_bases_cuda(
                    points: *const Affine<$curve>,
                    bases_size: i32,
                    precompute_factor: i32,
                    _c: i32,
                    are_bases_on_device: bool,
                    ctx: &DeviceContext,
                    output_bases: *mut Affine<$curve>,
                ) -> CudaError;
            }
        }

        impl MSM<$curve> for $curve {
            fn msm_unchecked(
                scalars: &(impl HostOrDeviceSlice<<$curve as Curve>::ScalarField> + ?Sized),
                points: &(impl HostOrDeviceSlice<Affine<$curve>> + ?Sized),
                cfg: &MSMConfig,
                results: &mut (impl HostOrDeviceSlice<Projective<$curve>> + ?Sized),
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
                points: &(impl HostOrDeviceSlice<Affine<$curve>> + ?Sized),
                precompute_factor: i32,
                _c: i32,
                ctx: &DeviceContext,
                output_bases: &mut DeviceSlice<Affine<$curve>>,
            ) -> IcicleResult<()> {
                unsafe {
                    $curve_prefix_indent::precompute_bases_cuda(
                        points.as_ptr(),
                        points.len() as i32,
                        precompute_factor,
                        _c,
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

#[macro_export]
macro_rules! impl_msm_bench {
    (
      $field_prefix:literal,
      $curve:ident
    ) => {
        use criterion::criterion_group;
        use criterion::criterion_main;
        use criterion::Criterion;
        use icicle_core::curve::Affine;
        use icicle_core::curve::Curve;
        use icicle_core::curve::Projective;
        use icicle_core::msm::msm;
        use icicle_core::msm::MSMConfig;
        use icicle_core::msm::MSM;
        use icicle_core::traits::FieldImpl;
        use icicle_core::traits::GenerateRandom;
        use icicle_cuda_runtime::device::warmup;
        use icicle_cuda_runtime::memory::DeviceVec;
        use icicle_cuda_runtime::memory::HostOrDeviceSlice;
        use icicle_cuda_runtime::memory::HostSlice;

        fn msm_for_bench<C: Curve + MSM<C>>(
            scalars_h: &(impl HostOrDeviceSlice<C::ScalarField> + ?Sized),
            precomputed_points_d: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
            cfg: &MSMConfig,
            msm_results: &mut (impl HostOrDeviceSlice<Projective<C>> + ?Sized),
            _seed: u32,
        ) {
            msm(scalars_h, precomputed_points_d, &cfg, msm_results).unwrap();
        }

        fn check_msm_batch<C: Curve + MSM<C>>(c: &mut Criterion)
        where
            <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
        {
            use criterion::black_box;
            use criterion::SamplingMode;
            use std::env;

            let group_id = format!("{} MSM ", $field_prefix);
            let mut group = c.benchmark_group(&group_id);
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);

            use icicle_core::msm::precompute_bases;
            use icicle_core::msm::tests::generate_random_affine_points_with_zeroes;
            use icicle_cuda_runtime::stream::CudaStream;

            const MAX_LOG2: u32 = 25; // max length = 2 ^ MAX_LOG2

            let max_log2 = env::var("MAX_LOG2")
                .unwrap_or_else(|_| MAX_LOG2.to_string())
                .parse::<u32>()
                .unwrap_or(MAX_LOG2);

            let stream = CudaStream::create().unwrap();
            let mut cfg = MSMConfig::default();
            cfg.ctx
                .stream = &stream;
            cfg.is_async = true;
            cfg.large_bucket_factor = 5;
            cfg.c = 4;

            warmup(&stream).unwrap();

            for test_size_log2 in (13u32..max_log2 + 1) {
                let test_size = 1 << test_size_log2;

                let points = generate_random_affine_points_with_zeroes(test_size, 10);
                for precompute_factor in [1, 4, 8] {
                    let mut precomputed_points_d = DeviceVec::cuda_malloc(precompute_factor * test_size).unwrap();
                    precompute_bases(
                        HostSlice::from_slice(&points),
                        precompute_factor as i32,
                        0,
                        &cfg.ctx,
                        &mut precomputed_points_d,
                    )
                    .unwrap();
                    for batch_size_log2 in [0, 4, 7] {
                        let batch_size = 1 << batch_size_log2;
                        let full_size = batch_size * test_size;

                        if full_size > 1 << max_log2 {
                            continue;
                        }

                        let mut scalars = <C::ScalarField as FieldImpl>::Config::generate_random(full_size);
                        let scalars = <C::ScalarField as FieldImpl>::Config::generate_random(full_size);
                        // a version of batched msm without using `cfg.points_size`, requires copying bases
                        let points_cloned: Vec<Affine<C>> = std::iter::repeat(points.clone())
                            .take(batch_size)
                            .flatten()
                            .collect();
                        let scalars_h = HostSlice::from_slice(&scalars);

                        let mut msm_results = DeviceVec::<Projective<C>>::cuda_malloc(batch_size).unwrap();
                        let mut points_d = DeviceVec::<Affine<C>>::cuda_malloc(full_size).unwrap();
                        points_d
                            .copy_from_host_async(HostSlice::from_slice(&points_cloned), &stream)
                            .unwrap();

                        cfg.precompute_factor = precompute_factor as i32;

                        let bench_descr = format!(
                            " {} x {} with precomp = {:?}",
                            test_size, batch_size, precompute_factor
                        );
                        group.bench_function(&bench_descr, |b| {
                            b.iter(|| {
                                msm_for_bench(
                                    scalars_h,
                                    &precomputed_points_d[..],
                                    &cfg,
                                    &mut msm_results[..],
                                    black_box(1),
                                )
                            })
                        });

                        stream
                            .synchronize()
                            .unwrap();
                    }
                }
            }
            stream
                .destroy()
                .unwrap();
        }

        criterion_group!(benches, check_msm_batch<$curve>);
        criterion_main!(benches);
    };
}
