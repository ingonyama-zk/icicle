use crate::{
    curve::{Affine, Curve},
    traits::FieldImpl,
};
use icicle_runtime::errors::eIcicleError;

#[doc(hidden)]
pub mod tests;

pub trait Pairing<C1: Curve, C2: Curve, F: FieldImpl> {
    fn pairing(p: &Affine<C1>, q: &Affine<C2>) -> Result<F, eIcicleError>;
}

pub fn pairing<C1, C2, F>(p: &Affine<C1>, q: &Affine<C2>) -> Result<F, eIcicleError>
where
    C1: Curve,
    C2: Curve,
    F: FieldImpl,
    C1: Pairing<C1, C2, F>,
{
    println!("SP: entered pairings");
    let result = C1::pairing(p, q);
    match result {
        Ok(val) => {
            println!("Pairing val: {:?}", val);
            Ok(val)
        },
        Err(err) => {
            println!("Pairing err: {:?}", err);
            Err(err)
        },
    }
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
            use super::{$curve, $curve_g2, $target_field, Affine};
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($curve_prefix, "_pairing")]
                pub(crate) fn pairing_ffi(
                    q: *const Affine<$curve>,
                    p: *const Affine<$curve_g2>,
                    out: *mut $target_field,
                ) -> eIcicleError;
            }
        }

        impl Pairing<$curve, $curve_g2, $target_field> for $curve {
            fn pairing(p: &Affine<$curve>, q: &Affine<$curve_g2>) -> Result<$target_field, eIcicleError> {
                //  Roman: icicle_core::pairing::check_pairing_bilinearity::<$curve, $curve_g2, $target_field>();
                let mut result = $target_field::zero();
                unsafe {
                    let pairing_cpp_res = $curve_prefix_ident::pairing_ffi(
                        p as *const Affine<$curve>,
                        q as *const Affine<$curve_g2>,
                        &mut result as *mut $target_field,
                    );
                    
                    //assert_eq!(pairing_cpp_res, eIcicleError::Success);

                    Ok(result)
                }
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
