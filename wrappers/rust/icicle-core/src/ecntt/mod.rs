#![cfg(feature = "ec_ntt")]
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::{
    curve::Curve,
    ntt::{FieldImpl, IcicleResult, NTTConfig, NTTDir},
};

pub use crate::curve::Projective;

// #[cfg(feature = "arkworks")] //TODO: uncomment on correctness test
#[doc(hidden)]
pub mod tests;

#[doc(hidden)]
pub trait ECNTT<C: Curve>: ECNTTUnchecked<Projective<C>, C::ScalarField> {}

#[doc(hidden)]
pub trait ECNTTUnchecked<T, F: FieldImpl> {
    fn ntt_unchecked(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<F>,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> IcicleResult<()>;

    fn ntt_inplace_unchecked(
        inout: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<F>,
    ) -> IcicleResult<()>;
}

#[macro_export]
macro_rules! impl_ecntt {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident,
        $curve:ident
    ) => {
        mod $field_prefix_ident {
            use crate::curve;
            use crate::curve::BaseCfg;
            use crate::ecntt::IcicleResult;
            use crate::ecntt::Projective;
            use crate::ecntt::{
                $curve, $field, $field_config, CudaError, DeviceContext, NTTConfig, NTTDir, DEFAULT_DEVICE_ID,
            };
            use icicle_core::ecntt::ECNTTUnchecked;
            use icicle_core::ecntt::ECNTT;
            use icicle_core::impl_ntt_without_domain;
            use icicle_core::ntt::NTT;
            use icicle_core::traits::IcicleResultWrap;
            use icicle_cuda_runtime::memory::HostOrDeviceSlice;

            pub type ProjectiveC = Projective<$curve>;
            impl_ntt_without_domain!(
                $field_prefix,
                $field,
                $field_config,
                ECNTTUnchecked,
                "_ecntt_",
                ProjectiveC
            );

            impl ECNTT<$curve> for $field_config {}
        }
    };
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
    input: &(impl HostOrDeviceSlice<Projective<C>> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<C::ScalarField>,
    output: &mut (impl HostOrDeviceSlice<Projective<C>> + ?Sized),
) -> IcicleResult<()>
where
    C::ScalarField: FieldImpl,
    <C::ScalarField as FieldImpl>::Config: ECNTT<C>,
{
    <<C::ScalarField as FieldImpl>::Config as ECNTTUnchecked<Projective<C>, C::ScalarField>>::ntt_unchecked(
        input, dir, &cfg, output,
    )
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
    inout: &mut (impl HostOrDeviceSlice<Projective<C>> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<C::ScalarField>,
) -> IcicleResult<()>
where
    C::ScalarField: FieldImpl,
    <C::ScalarField as FieldImpl>::Config: ECNTT<C>,
{
    <<C::ScalarField as FieldImpl>::Config as ECNTTUnchecked<Projective<C>, C::ScalarField>>::ntt_inplace_unchecked(
        inout, dir, &cfg,
    )
}

#[macro_export]
macro_rules! impl_ecntt_tests {
    (
      $field:ident,
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
            check_ecntt::<$curve>()
        }

        #[test]
        fn test_ecntt_batch() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ecntt_batch::<$curve>()
        }

        // #[test] //TODO: multi-device test
        // fn test_ntt_device_async() {
        //     // init_domain is in this test is performed per-device
        //     check_ecntt_device_async::<$field>()
        // }
    };
}

#[macro_export]
macro_rules! impl_ecntt_bench {
    (
      $field_prefix:literal,
      $field:ident,
      $curve:ident
    ) => {
        use icicle_core::ntt::ntt;
        use icicle_core::ntt::NTTDomain;
        use icicle_cuda_runtime::memory::HostOrDeviceSlice;
        use std::sync::OnceLock;

        use criterion::{black_box, criterion_group, criterion_main, Criterion};
        use icicle_core::{
            curve::Curve,
            ecntt::{ecntt, Projective},
            ntt::{FieldImpl, NTTConfig, NTTDir, NttAlgorithm, Ordering},
            traits::ArkConvertible,
        };

        use icicle_core::ecntt::ECNTT;
        use icicle_core::ntt::NTT;
        use icicle_cuda_runtime::memory::HostSlice;

        fn ecntt_for_bench<C: Curve>(
            points: &(impl HostOrDeviceSlice<Projective<C>> + ?Sized),
            mut batch_ntt_result: &mut (impl HostOrDeviceSlice<Projective<C>> + ?Sized),
            test_sizes: usize,
            batch_size: usize,
            is_inverse: NTTDir,
            ordering: Ordering,
            config: &mut NTTConfig<C::ScalarField>,
            _seed: u32,
        ) where
            C::ScalarField: ArkConvertible,
            <C::ScalarField as FieldImpl>::Config: ECNTT<C>,
            <C::ScalarField as FieldImpl>::Config: NTTDomain<C::ScalarField>,
        {
            ecntt(points, is_inverse, config, batch_ntt_result).unwrap();
        }

        static INIT: OnceLock<()> = OnceLock::new();

        fn benchmark_ecntt<C: Curve>(c: &mut Criterion)
        where
            C::ScalarField: ArkConvertible,
            <C::ScalarField as FieldImpl>::Config: ECNTT<C>,
            <C::ScalarField as FieldImpl>::Config: NTTDomain<C::ScalarField>,
        {
            use criterion::SamplingMode;
            use icicle_core::ntt::ntt;
            use icicle_core::ntt::tests::init_domain;
            use icicle_core::ntt::NTTDomain;
            use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;

            let group_id = format!("{} EC NTT ", $field_prefix);
            let mut group = c.benchmark_group(&group_id);
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);

            const MAX_LOG2: u32 = 9; // max length = 2 ^ MAX_LOG2 //TODO: should be limited by device ram only after fix

            let max_log2 = env::var("MAX_LOG2")
                .unwrap_or_else(|_| MAX_LOG2.to_string())
                .parse::<u32>()
                .unwrap_or(MAX_LOG2);

            const FAST_TWIDDLES_MODE: bool = false;

            INIT.get_or_init(move || init_domain::<$field>(1 << max_log2, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));

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
                    let mut batch_ntt_result = vec![Projective::<C>::zero(); full_size];
                    let batch_ntt_result = HostSlice::from_mut_slice(&mut batch_ntt_result);
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
                                let bench_descr = format!(
                                    "{:?} {:?} {:?} {} x {}",
                                    alg, ordering, is_inverse, test_size, batch_size
                                );
                                group.bench_function(&bench_descr, |b| {
                                    b.iter(|| {
                                        ecntt_for_bench::<C>(
                                            points,
                                            batch_ntt_result,
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

        criterion_group!(benches, benchmark_ecntt<$curve>);
        criterion_main!(benches);
    };
}
