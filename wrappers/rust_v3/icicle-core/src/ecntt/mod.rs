#![cfg(feature = "ec_ntt")]
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

pub use crate::curve::Projective;
use crate::{
    curve::Curve,
    ntt::{NTTConfig, NTTDir},
    traits::FieldImpl,
};

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
    ) -> Result<(), eIcicleError>;

    fn ntt_inplace_unchecked(
        inout: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<F>,
    ) -> Result<(), eIcicleError>;
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
            use crate::ecntt::Projective;
            use crate::ecntt::{$curve, $field, $field_config};
            use icicle_core::ecntt::{ECNTTUnchecked, ECNTT};
            use icicle_core::impl_ntt_without_domain;
            use icicle_core::ntt::{NTTConfig, NTTDir, NTTInitDomainConfig, NTT};
            use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

            pub type ProjectiveC = Projective<$curve>;
            impl_ntt_without_domain!(
                $field_prefix,
                $field,
                $field_config,
                ECNTTUnchecked,
                "_ecntt",
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
) -> Result<(), eIcicleError>
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
) -> Result<(), eIcicleError>
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
        pub mod test_ecntt {
            use super::*;
            use icicle_core::ntt::tests::init_domain;
            use icicle_core::test_utilities;
            use std::sync::Once;

            const MAX_SIZE: u64 = 1 << 18;
            static INIT: Once = Once::new();

            pub fn initialize() {
                INIT.call_once(move || {
                    test_utilities::test_load_and_init_devices();
                    // init domain for both devices
                    test_utilities::test_set_ref_device();
                    init_domain::<$field>(MAX_SIZE, false);

                    test_utilities::test_set_main_device();
                    init_domain::<$field>(MAX_SIZE, false);
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

// TODO Yuval : becnhmarks
