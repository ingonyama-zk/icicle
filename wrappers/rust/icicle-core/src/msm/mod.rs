use crate::projective::Projective;
use icicle_runtime::{
    config::ConfigExtension,
    errors::{eIcicleError, IcicleError},
    memory::{DeviceSlice, HostOrDeviceSlice},
    stream::IcicleStreamHandle,
};

#[doc(hidden)]
pub mod tests;

/// Struct that encodes MSM parameters to be passed into the [`msm`](msm) function.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MSMConfig {
    pub stream_handle: IcicleStreamHandle,

    /// The number of extra bases to pre-compute for each point. See the `precompute_bases` function, `precompute_factor` passed
    /// there needs to be equal to the one used here. Larger values decrease the number of computations
    /// to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute).
    ///
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

    pub batch_size: i32,
    pub are_points_shared_in_batch: bool,
    /// MSMs in batch share the bases. If false, expecting #bases==#scalars
    are_scalars_on_device: bool,
    pub are_scalars_montgomery_form: bool,
    are_bases_on_device: bool,
    pub are_bases_montgomery_form: bool,
    are_results_on_device: bool,

    /// Whether to run the MSM asynchronously. If set to `true`, the MSM function will be non-blocking
    /// and you'd need to synchronize it explicitly by running `stream.synchronize()`
    /// If set to `false`, the MSM function will block the current CPU thread.
    pub is_async: bool,
    pub ext: ConfigExtension,
}

// backend specific options
pub const CUDA_MSM_LARGE_BUCKET_FACTOR: &str = "large_bucket_factor";
pub const CUDA_MSM_IS_BIG_TRIANGLE: &str = "is_big_triangle";

impl Default for MSMConfig {
    fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            precompute_factor: 1,
            c: 0,
            bitsize: 0,
            batch_size: 1,
            are_points_shared_in_batch: true,
            are_scalars_on_device: false,
            are_scalars_montgomery_form: false,
            are_bases_on_device: false,
            are_bases_montgomery_form: false,
            are_results_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[doc(hidden)]
pub trait MSM<P: Projective> {
    fn msm_unchecked(
        scalars: &(impl HostOrDeviceSlice<P::ScalarField> + ?Sized),
        bases: &(impl HostOrDeviceSlice<P::Affine> + ?Sized),
        cfg: &MSMConfig,
        results: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    ) -> Result<(), IcicleError>;

    fn precompute_bases_unchecked(
        bases: &(impl HostOrDeviceSlice<P::Affine> + ?Sized),
        cfg: &MSMConfig,
        output_bases: &mut DeviceSlice<P::Affine>,
    ) -> Result<(), IcicleError>;

    fn msm(
        scalars: &(impl HostOrDeviceSlice<P::ScalarField> + ?Sized),
        bases: &(impl HostOrDeviceSlice<P::Affine> + ?Sized),
        cfg: &MSMConfig,
        results: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    ) -> Result<(), IcicleError> {
        if bases.len() % (cfg.precompute_factor as usize) != 0 {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "Precompute factor {} does not divide the number of bases {}",
                    cfg.precompute_factor,
                    bases.len()
                ),
            ));
        }
        let bases_size = bases.len() / (cfg.precompute_factor as usize);
        if scalars.len() % bases_size != 0 {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "Number of bases {} does not divide the number of scalars {}",
                    bases_size,
                    scalars.len()
                ),
            ));
        }
        if scalars.len() % results.len() != 0 {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "Number of results {} does not divide the number of scalars {}",
                    results.len(),
                    scalars.len()
                ),
            ));
        }

        // check device slices are on active device
        if scalars.is_on_device() && !scalars.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "scalars not allocated on an inactive device",
            ));
        }
        if bases.is_on_device() && !bases.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "bases not allocated on an inactive device",
            ));
        }
        if results.is_on_device() && !results.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "results not allocated on an inactive device",
            ));
        }

        let mut local_cfg = cfg.clone();
        local_cfg.are_points_shared_in_batch = bases_size < scalars.len();
        local_cfg.batch_size = results.len() as i32;
        local_cfg.are_scalars_on_device = scalars.is_on_device();
        local_cfg.are_bases_on_device = bases.is_on_device();
        local_cfg.are_results_on_device = results.is_on_device();

        Self::msm_unchecked(scalars, bases, &local_cfg, results)
    }

    fn precompute_bases(
        points: &(impl HostOrDeviceSlice<P::Affine> + ?Sized),
        config: &MSMConfig,
        output_bases: &mut DeviceSlice<P::Affine>,
    ) -> Result<(), IcicleError> {
        if output_bases.len() != points.len() * (config.precompute_factor as usize) {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "Precompute factor is probably incorrect: expected {} but got {}",
                    output_bases.len() / points.len(),
                    config.precompute_factor
                ),
            ));
        }

        if !output_bases.is_on_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "output_bases not allocated on a device",
            ));
        }

        Self::precompute_bases_unchecked(points, config, output_bases)
    }
}

/// Computes the multi-scalar multiplication, or MSM: `s1*P1 + s2*P2 + ... + sn*Pn`, or a batch of several MSMs.
///
/// # Arguments
///
/// * `scalars` - scalar values `s1, s2, ..., sn`.
///
/// * `bases` - bases `P1, P2, ..., Pn`. The number of bases can be smaller than the number of scalars
/// in the case of batch MSM. In this case bases are reused periodically. Alternatively, there can be more bases
/// than scalars if precomputation has been performed, you need to set `cfg.precompute_factor` in that case.
///
/// * `cfg` - config used to specify extra arguments of the MSM.
///
/// * `results` - buffer to write results into. Its length is equal to the batch size i.e. number of MSMs to compute.
///
/// Returns `Ok(())` if no errors occurred or an `IcicleError` otherwise.
pub fn msm<P: Projective + MSM<P>>(
    scalars: &(impl HostOrDeviceSlice<P::ScalarField> + ?Sized),
    bases: &(impl HostOrDeviceSlice<P::Affine> + ?Sized),
    cfg: &MSMConfig,
    results: &mut (impl HostOrDeviceSlice<P> + ?Sized),
) -> Result<(), IcicleError> {
    P::msm(scalars, bases, cfg, results)
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
/// Returns `Ok(())` if no errors occurred or an `IcicleError` otherwise.
pub fn precompute_bases<P: Projective + MSM<P>>(
    points: &(impl HostOrDeviceSlice<P::Affine> + ?Sized),
    config: &MSMConfig,
    output_bases: &mut DeviceSlice<P::Affine>,
) -> Result<(), IcicleError> {
    P::precompute_bases(points, config, output_bases)
}

#[macro_export]
macro_rules! impl_msm {
    (
      $curve_prefix:literal,
      $curve_prefix_ident:ident,
      $projective_type:ident
    ) => {
        mod $curve_prefix_ident {
            use crate::msm::$projective_type;
            use icicle_core::projective::Projective;
            use icicle_core::msm::MSMConfig;
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($curve_prefix, "_msm")]
                pub(crate) fn msm_ffi(
                    scalars: *const <$projective_type as Projective>::ScalarField,
                    points: *const <$projective_type as Projective>::Affine,
                    count: i32,
                    config: &MSMConfig,
                    out: *mut $projective_type,
                ) -> eIcicleError;

                #[link_name = concat!($curve_prefix, "_msm_precompute_bases")]
                pub(crate) fn precompute_bases_ffi(
                    points: *const <$projective_type as Projective>::Affine,
                    bases_size: i32,
                    config: &MSMConfig,
                    output_bases: *mut <$projective_type as Projective>::Affine,
                ) -> eIcicleError;
            }
        }

        impl MSM<$projective_type> for $projective_type {
            fn msm_unchecked(
                scalars: &(impl HostOrDeviceSlice<<$projective_type as icicle_core::projective::Projective>::ScalarField> + ?Sized),
                points: &(impl HostOrDeviceSlice<<$projective_type as icicle_core::projective::Projective>::Affine> + ?Sized),
                cfg: &MSMConfig,
                results: &mut (impl HostOrDeviceSlice<$projective_type> + ?Sized),
            ) -> Result<(), IcicleError> {
                unsafe {
                    $curve_prefix_ident::msm_ffi(
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
                points: &(impl HostOrDeviceSlice<<$projective_type as icicle_core::projective::Projective>::Affine> + ?Sized),
                config: &MSMConfig,
                output_bases: &mut DeviceSlice<<$projective_type as icicle_core::projective::Projective>::Affine>,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $curve_prefix_ident::precompute_bases_ffi(
                        points.as_ptr(),
                        points.len() as i32 / config.batch_size,
                        config,
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
        use icicle_runtime::test_utilities;
        pub fn initialize() {
            test_utilities::test_load_and_init_devices();
            test_utilities::test_set_main_device();
        }

        #[test]
        fn test_msm() {
            initialize();
            check_msm::<$curve>();
        }

        #[test]
        fn test_msm_batch_shared() {
            initialize();
            check_msm_batch_shared::<$curve>()
        }

        #[test]
        fn test_msm_batch_not_shared() {
            initialize();
            check_msm_batch_not_shared::<$curve>()
        }

        #[test]
        fn test_msm_skewed_distributions() {
            initialize();
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
        use criterion::{criterion_group, criterion_main, Criterion};
        use icicle_core::msm::{msm, MSMConfig, CUDA_MSM_LARGE_BUCKET_FACTOR, MSM};
        use icicle_core::projective::Projective;
        use icicle_core::traits::GenerateRandom;
        use icicle_runtime::{
            device::Device,
            get_active_device, is_device_available,
            memory::{DeviceVec, HostOrDeviceSlice, IntoIcicleSlice, IntoIcicleSliceMut},
            runtime::{load_backend_from_env_or_default, warmup},
            set_device,
            stream::IcicleStream,
        };
        use std::env;

        fn load_and_init_backend_device() {
            // Attempt to load the backends
            let _ = load_backend_from_env_or_default(); // try loading from /opt/icicle/backend or env ${ICICLE_BACKEND_INSTALL_DIR}

            // Check if BENCH_TARGET is defined
            let target = env::var("BENCH_TARGET").unwrap_or_else(|_| {
                // If not defined, try CUDA first, fallback to CPU
                if is_device_available(&Device::new("CUDA", 0)) {
                    "CUDA".to_string()
                } else {
                    "CPU".to_string()
                }
            });

            // Initialize the device with the determined target
            let device = Device::new(&target, 0);
            set_device(&device).unwrap();

            println!("ICICLE benchmark with {:?}", device);
        }

        fn check_msm_batch<P: Projective + MSM<P>>(c: &mut Criterion)
        where
            P::ScalarField: GenerateRandom,
        {
            use criterion::black_box;
            use criterion::SamplingMode;
            use icicle_core::msm::precompute_bases;
            use icicle_core::msm::tests::generate_random_affine_points_with_zeroes;

            load_and_init_backend_device();

            let group_id = format!("{} MSM ", $field_prefix);
            let mut group = c.benchmark_group(&group_id);
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);

            const MIN_LOG2: u32 = 13; // min msm length = 2 ^ MIN_LOG2
            const MAX_LOG2: u32 = 25; // max msm length = 2 ^ MAX_LOG2

            let min_log2 = env::var("MIN_LOG2")
                .unwrap_or_else(|_| MIN_LOG2.to_string())
                .parse::<u32>()
                .unwrap_or(MIN_LOG2);

            let max_log2 = env::var("MAX_LOG2")
                .unwrap_or_else(|_| MAX_LOG2.to_string())
                .parse::<u32>()
                .unwrap_or(MAX_LOG2);

            let mut stream = IcicleStream::create().unwrap();
            let mut cfg = MSMConfig::default();
            cfg.stream_handle = *stream;
            cfg.is_async = true;
            cfg.ext
                .set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, 5);
            cfg.c = 4;

            warmup(&stream).unwrap();

            for test_size_log2 in (min_log2..=max_log2) {
                let test_size = 1 << test_size_log2;

                let points = generate_random_affine_points_with_zeroes::<P::Affine>(test_size, 10);
                for precompute_factor in [1, 4, 8] {
                    let mut precomputed_points_d = DeviceVec::<P::Affine>::malloc(precompute_factor * test_size);
                    cfg.precompute_factor = precompute_factor as i32;
                    precompute_bases::<P>(points.into_slice(), &cfg, &mut precomputed_points_d).unwrap();
                    for batch_size_log2 in [0, 4, 7] {
                        let batch_size = 1 << batch_size_log2;
                        let full_size = batch_size * test_size;

                        if full_size > 1 << max_log2 {
                            continue;
                        }

                        let mut scalars = P::ScalarField::generate_random(full_size);
                        let scalars = P::ScalarField::generate_random(full_size);
                        // a version of batched msm without using `cfg.points_size`, requires copying bases

                        let scalars_h = scalars.into_slice();

                        let mut msm_results = DeviceVec::<P>::malloc(batch_size);

                        let bench_descr = format!(
                            " {} x {} with precomp = {:?}",
                            test_size, batch_size, precompute_factor
                        );

                        group.bench_function(&bench_descr, |b| {
                            b.iter(|| {
                                msm(
                                    scalars_h,
                                    precomputed_points_d.into_slice(),
                                    &cfg,
                                    msm_results.into_slice_mut(),
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
