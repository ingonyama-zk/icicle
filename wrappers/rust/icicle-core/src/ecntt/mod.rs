use icicle_runtime::{memory::HostOrDeviceSlice, IcicleError};

pub use crate::projective::Projective;
use crate::{
    curve::Curve,
    field::Field,
    ntt::{NTTConfig, NTTDir},
};

#[doc(hidden)]
pub mod tests;

#[doc(hidden)]
pub trait ECNTT<Point: Projective, Scalar: Field> {
    fn ntt_unchecked(
        input: &(impl HostOrDeviceSlice<Point> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<Scalar>,
        output: &mut (impl HostOrDeviceSlice<Point> + ?Sized),
    ) -> Result<(), IcicleError>;

    fn ntt_inplace_unchecked(
        inout: &mut (impl HostOrDeviceSlice<Point> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<Scalar>,
    ) -> Result<(), IcicleError>;

    fn ntt(
        input: &(impl HostOrDeviceSlice<Point> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<Scalar>,
        output: &mut (impl HostOrDeviceSlice<Point> + ?Sized),
    ) -> Result<(), IcicleError> {
        Self::ntt_unchecked(input, dir, cfg, output)
    }

    fn ntt_inplace(
        inout: &mut (impl HostOrDeviceSlice<Point> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<Scalar>,
    ) -> Result<(), IcicleError> {
        Self::ntt_inplace_unchecked(inout, dir, cfg)
    }
}

/// Computes the ECNTT, or a batch of several ECNTTs.
///
/// # Arguments
///
/// * `input` - inputs of the ECNTT.
///
/// * `dir` - whether to compute forward of inverse ECNTT.
///
/// * `cfg` - config used to specify extra arguments of the ECNTT.
///
/// * `output` - buffer to write the ECNTT outputs into. Must be of the same size as `input`.
pub fn ecntt<C: Curve>(
    input: &(impl HostOrDeviceSlice<C::Projective> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<C::ScalarField>,
    output: &mut (impl HostOrDeviceSlice<C::Projective> + ?Sized),
) -> Result<(), IcicleError>
where
    C::ScalarField: ECNTT<C::Projective, C::ScalarField>,
{
    <C::ScalarField as ECNTT<C::Projective, C::ScalarField>>::ntt(input, dir, cfg, output)
}

/// Computes the ECNTT, or a batch of several ECNTTs inplace.
///
/// # Arguments
///
/// * `inout` - buffer with inputs to also write the ECNTT outputs into.
///
/// * `dir` - whether to compute forward of inverse ECNTT.
///
/// * `cfg` - config used to specify extra arguments of the ECNTT.
pub fn ecntt_inplace<C: Curve>(
    inout: &mut (impl HostOrDeviceSlice<C::Projective> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<C::ScalarField>,
) -> Result<(), IcicleError>
where
    C::ScalarField: ECNTT<C::Projective, C::ScalarField>,
{
    <C::ScalarField as ECNTT<C::Projective, C::ScalarField>>::ntt_inplace(inout, dir, &cfg)
}

#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! impl_ecntt {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $curve:ident
    ) => {
        mod $field_prefix_ident {
            use crate::ecntt::{$curve, $field};
            use icicle_core::curve::Curve;
            use icicle_core::ecntt::ECNTT;
            use icicle_core::impl_ntt_without_domain;
            use icicle_core::ntt::{NTTConfig, NTTDir, NTTInitDomainConfig, NTT};
            use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, IcicleError};

            type ProjectiveType = <$curve as Curve>::Projective;
            impl_ntt_without_domain!($field_prefix, $field, ECNTT, "_ecntt", ProjectiveType);
        }
    };
}

#[macro_export]
macro_rules! impl_ecntt_tests {
    (
      $curve:ident
    ) => {
        pub mod test_ecntt {
            use super::*;
            use icicle_core::ntt::tests::init_domain;
            use icicle_runtime::test_utilities;
            use std::sync::Once;

            const MAX_SIZE: u64 = 1 << 18;
            static INIT: Once = Once::new();

            pub fn initialize() {
                INIT.call_once(move || {
                    test_utilities::test_load_and_init_devices();
                    // init domain for both devices
                    test_utilities::test_set_ref_device();
                    init_domain::<<$curve as icicle_core::curve::Curve>::ScalarField>(MAX_SIZE, false);

                    test_utilities::test_set_main_device();
                    init_domain::<<$curve as icicle_core::curve::Curve>::ScalarField>(MAX_SIZE, false);
                });
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_ecntt() {
                initialize();
                check_ecntt::<$curve>()
            }

            #[test]
            fn test_ecntt_batch() {
                initialize();
                check_ecntt_batch::<$curve>()
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ecntt_bench {
    (
      $field_prefix:literal,
      $field:ident,
      $curve:ident
    ) => {
        use criterion::{black_box, criterion_group, criterion_main, Criterion};
        use icicle_core::projective::Projective;
        use icicle_core::{
            curve::Curve,
            ecntt::{ecntt, ECNTT},
            ntt::{ntt, NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NttAlgorithm, Ordering, NTT},
            traits::GenerateRandom,
            vec_ops::VecOps,
        };
        use icicle_runtime::{
            device::Device,
            get_active_device, is_device_available,
            memory::{HostOrDeviceSlice, HostSlice},
            runtime::load_backend_from_env_or_default,
            set_device,
        };
        use std::{env, sync::OnceLock};

        static INIT: OnceLock<()> = OnceLock::new();

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

        fn benchmark_ecntt<C: Curve>(c: &mut Criterion)
        where
            C::ScalarField: ECNTT<C::Projective, C::ScalarField>,
        {
            use criterion::SamplingMode;
            use icicle_core::ntt::tests::init_domain;
            use std::env;

            load_and_init_backend_device();

            let group_id = format!("{} EC NTT ", $field_prefix);
            let mut group = c.benchmark_group(&group_id);
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);

            const MAX_LOG2: u32 = 9; // max length = 2 ^ MAX_LOG2 //TODO: should be limited by device ram

            let max_log2 = env::var("MAX_LOG2")
                .unwrap_or_else(|_| MAX_LOG2.to_string())
                .parse::<u32>()
                .unwrap_or(MAX_LOG2);

            const FAST_TWIDDLES_MODE: bool = false;

            INIT.get_or_init(move || init_domain::<$field>(1 << max_log2, FAST_TWIDDLES_MODE));

            for test_size_log2 in [4, 8] {
                for batch_size_log2 in [1, 1 << 4, 128] {
                    let test_size = 1 << test_size_log2;
                    let batch_size = 1 << batch_size_log2;
                    let full_size = batch_size * test_size;

                    if full_size > 1 << max_log2 {
                        continue;
                    }

                    let points = C::generate_random_projective_points(test_size);
                    let points = HostSlice::from_slice(&points);
                    let mut batch_ntt_result = vec![C::Projective::zero(); full_size];
                    let batch_ntt_result = HostSlice::from_mut_slice(&mut batch_ntt_result);
                    let mut config = NTTConfig::<C::ScalarField>::default();
                    config.ordering = Ordering::kNN;
                    config.batch_size = batch_size as i32;
                    for dir in [NTTDir::kForward, NTTDir::kInverse] {
                        let bench_descr = format!("{:?} {:?} {} x {}", config.ordering, dir, test_size, batch_size);
                        group.bench_function(&bench_descr, |b| {
                            b.iter(|| ecntt::<C>(points, dir, &mut config, batch_ntt_result))
                        });
                    }
                }
            }

            group.finish();
        }

        criterion_group!(benches, benchmark_ecntt<$curve>);
        criterion_main!(benches);
    };
}
