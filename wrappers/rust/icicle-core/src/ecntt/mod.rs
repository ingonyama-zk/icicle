use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::{
    curve::Curve,
    ntt::{FieldImpl, IcicleResult, NTTConfig, NTTDir},
};

pub use crate::curve::Projective;

#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;

#[doc(hidden)]
pub trait ECNTT<C: Curve> {
    fn ecntt_unchecked(
        input: &HostOrDeviceSlice<Projective<C>>,
        dir: NTTDir,
        cfg: &NTTConfig<C::ScalarField>,
        output: &mut HostOrDeviceSlice<Projective<C>>,
    ) -> IcicleResult<()>;
    // fn initialize_domain(primitive_root: C::ScalarField, ctx: &DeviceContext) -> IcicleResult<()>;
    // fn initialize_domain_fast_twiddles_mode(primitive_root: C::ScalarField, ctx: &DeviceContext) -> IcicleResult<()>;
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
    input: &HostOrDeviceSlice<Projective<C>>,
    dir: NTTDir,
    cfg: &NTTConfig<C::ScalarField>,
    output: &mut HostOrDeviceSlice<Projective<C>>,
) -> IcicleResult<()>
where
    <C::BaseField as FieldImpl>::Config: ECNTT<C>,
{
    if input.len() != output.len() {
        panic!(
            "input and output lengths {}; {} do not match",
            input.len(),
            output.len()
        );
    }
    let mut local_cfg = cfg.clone();
    local_cfg.are_inputs_on_device = input.is_on_device();
    local_cfg.are_outputs_on_device = output.is_on_device();
    <C::BaseField as FieldImpl>::Config::ecntt_unchecked(input, dir, &local_cfg, output)
}

#[macro_export]
macro_rules! impl_ecntt {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident,
        $base_field:ident,
        $curve:ident
    ) => {
        mod $field_prefix_ident {
            use crate::ecntt::Projective;
            use crate::ecntt::{
                $base_field, $curve, $field, $field_config, CudaError, DeviceContext, NTTConfig, NTTDir,
                DEFAULT_DEVICE_ID,
            };

            extern "C" {
                #[link_name = concat!($field_prefix, "ECNTTCuda")]
                pub(crate) fn ecntt_cuda(
                    input: *const Projective<$curve>,
                    size: i32,
                    dir: NTTDir,
                    config: &NTTConfig<$field>,
                    output: *mut Projective<$curve>,
                ) -> CudaError;
            }
        }
        //<C::ScalarField as FieldImpl>::Config
        impl ECNTT<$curve> for $base_field {
            fn ecntt_unchecked(
                input: &HostOrDeviceSlice<Projective<$curve>>,
                dir: NTTDir,
                cfg: &NTTConfig<$field>,
                output: &mut HostOrDeviceSlice<Projective<$curve>>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::ecntt_cuda(
                        input.as_ptr(),
                        (input.len() / (cfg.batch_size as usize)) as i32,
                        dir,
                        cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ecntt_tests {
    (
      $field:ident,
      $base_field:ident,
      $curve:ident
    ) => {
        use icicle_core::ntt::tests::init_domain;
        use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
        const MAX_SIZE: u64 = 1 << 18;
        static INIT: OnceLock<()> = OnceLock::new();
        const FAST_TWIDDLES_MODE: bool = false;

        #[test]
        fn test_ecntt() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ecntt::<$field, $base_field, $curve>()
        }

        #[test]
        fn test_ecntt_batch() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ecntt_batch::<$field, $base_field, $curve>()
        }

        // #[test]
        // fn test_ntt_device_async() {
        //     // init_domain is in this test is performed per-device
        //     check_ntt_device_async::<$field>()
        // }
    };
}

#[macro_export]
macro_rules! impl_ecntt_bench {
    (
      $field_prefix:literal,
      $field:ident,
      $base_field:ident,
      $curve:ident
    ) => {
        use std::sync::OnceLock;

        #[cfg(feature = "arkworks")]
        use ark_ff::FftField;

        use criterion::{black_box, criterion_group, criterion_main, Criterion};
        use icicle_core::{
            traits::ArkConvertible,
            curve::Curve,
            ecntt::{ecntt, Projective},
            ntt::{FieldImpl, HostOrDeviceSlice, NTTConfig, NTTDir, NttAlgorithm, Ordering},
        };

        use icicle_core::ecntt::ECNTT;
        use icicle_core::ntt::NTT;

        fn ecntt_for_bench<F: FieldImpl + ArkConvertible, C: Curve>(
            points: &HostOrDeviceSlice<Projective<C>>,
            mut batch_ntt_result: &mut HostOrDeviceSlice<Projective<C>>,
            test_sizes: usize,
            batch_size: usize,
            is_inverse: NTTDir,
            ordering: Ordering,
            config: &mut NTTConfig<C::ScalarField>,
            _seed: u32,
        ) where
            <C::BaseField as FieldImpl>::Config: ECNTT<C>,
        {
            ecntt(&points, is_inverse, config, &mut batch_ntt_result).unwrap();
        }

        static INIT: OnceLock<()> = OnceLock::new();

        fn benchmark_ecntt<F: FieldImpl + ArkConvertible, C: Curve>(c: &mut Criterion)
        where
            F::ArkEquivalent: FftField,
            <F as FieldImpl>::Config: NTT<F>,
            <C::BaseField as FieldImpl>::Config: ECNTT<C>,
        {
            use icicle_core::ntt::tests::init_domain;
            use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
            use criterion::SamplingMode;

            let group_id = format!("{} EC NTT", $field_prefix);
            let mut group = c.benchmark_group(&group_id);
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);

            const MAX_SIZE: u64 = 1 << 18;
            const FAST_TWIDDLES_MODE: bool = false;
            INIT.get_or_init(move || init_domain::<F>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));

            let test_sizes = [1 << 4, 1 << 8];
            let batch_sizes = [1, 1 << 4, 128];
            for test_size in test_sizes {
                for batch_size in batch_sizes {
                    let points = HostOrDeviceSlice::on_host(C::generate_random_projective_points(test_size * batch_size));
                    let mut batch_ntt_result = HostOrDeviceSlice::on_host(vec![Projective::zero(); batch_size * test_size]);
                    let mut config = NTTConfig::default();
                    for is_inverse in [NTTDir::kInverse, NTTDir::kForward] {
                        for ordering in [
                            Ordering::kNN,
                            // Ordering::kNR, // times are ~ same as kNN
                            // Ordering::kRN,
                            // Ordering::kRR,
                            // Ordering::kNM, // no mixed radix ecntt
                            // Ordering::kMN,
                        ] {
                            config.ordering = ordering;
                            for alg in [NttAlgorithm::Radix2] {
                                config.batch_size = batch_size as i32;
                                config.ntt_algorithm = alg;
                                let bench_descr = format!("{:?} {:?} {:?} {} x {}", alg, ordering, is_inverse, test_size, batch_size);
                                group.bench_function(&bench_descr, |b| {
                                    b.iter(|| {
                                        ecntt_for_bench::<F, C>(
                                            &points,
                                            &mut batch_ntt_result,
                                            test_size,
                                            batch_size,
                                            is_inverse,
                                            ordering,
                                            &mut config,
                                            black_box(1),
                                        )
                                    })
                                });
                            }
                        }
                    }
                }
            }

            group.finish();
        }

        criterion_group!(benches, benchmark_ecntt<$field, $curve>);
        criterion_main!(benches);
    };
}
