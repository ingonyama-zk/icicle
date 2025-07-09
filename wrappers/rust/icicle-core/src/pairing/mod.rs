use crate::{field::Field, projective::Projective};
use icicle_runtime::IcicleError;

#[doc(hidden)]
pub mod tests;

pub trait Pairing<P1: Projective, P2: Projective, F: Field> {
    fn pairing(p: &P1::Affine, q: &P2::Affine) -> Result<F, IcicleError>;
}

pub fn pairing<P1, P2, F>(p: &P1::Affine, q: &P2::Affine) -> Result<F, IcicleError>
where
    P1: Projective,
    P2: Projective,
    F: Field,
    P1: Pairing<P1, P2, F>,
{
    P1::pairing(p, q)
}

#[macro_export]
macro_rules! impl_pairing {
    (
      $curve_prefix:literal,
      $curve_prefix_ident:ident,
      $projective_type:ident,
      $projective_type_g2:ident,
      $target_field:ident
    ) => {
        mod $curve_prefix_ident {
            use super::{$projective_type, $projective_type_g2, $target_field};
            use icicle_core::projective::Projective;
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($curve_prefix, "_pairing")]
                pub(crate) fn pairing_ffi(
                    q: *const <$projective_type as Projective>::Affine,
                    p: *const <$projective_type_g2 as Projective>::Affine,
                    out: *mut $target_field,
                ) -> eIcicleError;
            }
        }

        impl Pairing<$projective_type, $projective_type_g2, $target_field> for $projective_type {
            fn pairing(
                p: &<$projective_type as icicle_core::projective::Projective>::Affine,
                q: &<$projective_type_g2 as icicle_core::projective::Projective>::Affine,
            ) -> Result<$target_field, IcicleError> {
                let mut result = $target_field::zero();
                unsafe {
                    $curve_prefix_ident::pairing_ffi(
                        p as *const <$projective_type as icicle_core::projective::Projective>::Affine,
                        q as *const <$projective_type_g2 as icicle_core::projective::Projective>::Affine,
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
      $projective_type:ident,
      $projective_type_g2:ident,
      $target_field:ident
    ) => {
        #[test]
        fn test_pairing_bilinearity() {
            check_pairing_bilinearity::<$projective_type, $projective_type_g2, $target_field>();
        }
    };
}
