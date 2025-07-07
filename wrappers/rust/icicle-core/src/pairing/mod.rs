use crate::{curve::Curve, field::Field};
use icicle_runtime::IcicleError;

#[doc(hidden)]
pub mod tests;

pub trait Pairing<C1: Curve, C2: Curve, F: Field> {
    fn pairing(p: &C1::Affine, q: &C2::Affine) -> Result<F, IcicleError>;
}

pub fn pairing<C1, C2, F>(p: &C1::Affine, q: &C2::Affine) -> Result<F, IcicleError>
where
    C1: Curve,
    C2: Curve,
    F: Field,
    C1: Pairing<C1, C2, F>,
{
    C1::pairing(p, q)
}

#[macro_export]
macro_rules! impl_pairing {
    (
      $curve_prefix:literal,
      $curve_prefix_ident:ident,
      $curve:ident,
      $curve_g2:ident,
      $target_field:ident
    ) => {
        mod $curve_prefix_ident {
            use super::{$curve, $curve_g2, $target_field};
            use icicle_core::curve::Curve;
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($curve_prefix, "_pairing")]
                pub(crate) fn pairing_ffi(
                    q: *const <$curve as Curve>::Affine,
                    p: *const <$curve_g2 as Curve>::Affine,
                    out: *mut $target_field,
                ) -> eIcicleError;
            }
        }

        impl Pairing<$curve, $curve_g2, $target_field> for $curve {
            fn pairing(
                p: &<$curve as icicle_core::curve::Curve>::Affine,
                q: &<$curve_g2 as icicle_core::curve::Curve>::Affine,
            ) -> Result<$target_field, IcicleError> {
                let mut result = $target_field::zero();
                unsafe {
                    $curve_prefix_ident::pairing_ffi(
                        p as *const <$curve as icicle_core::curve::Curve>::Affine,
                        q as *const <$curve_g2 as icicle_core::curve::Curve>::Affine,
                        &mut result as *mut $target_field,
                    )
                    .wrap()
                }
                .map(|_| result)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_pairing_tests {
    (
      $curve:ident,
      $curve_g2:ident,
      $target_field:ident
    ) => {
        #[test]
        fn test_pairing_bilinearity() {
            check_pairing_bilinearity::<$curve, $curve_g2, $target_field>();
        }
    };
}
