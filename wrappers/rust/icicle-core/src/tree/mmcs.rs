use icicle_cuda_runtime::memory::HostSlice;

use crate::{error::IcicleResult, ntt::FieldImpl};
use crate::{hash::SpongeHash, Matrix};

use super::TreeBuilderConfig;

pub trait FieldMmcs<F, Compression, Hasher>
where
    F: FieldImpl,
    Compression: SpongeHash<F, F>,
    Hasher: SpongeHash<F, F>,
{
    fn mmcs_commit(
        leaves: Vec<Matrix>,
        digests: &mut HostSlice<F>,
        hasher: &Hasher,
        compression: &Compression,
        config: &TreeBuilderConfig,
    ) -> IcicleResult<()>;
}

#[macro_export]
macro_rules! impl_mmcs {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident,
      $mmcs:ident
    ) => {
        mod $field_prefix_ident {
            use super::*;
            use icicle_cuda_runtime::error::CudaError;

            extern "C" {
                #[link_name = concat!($field_prefix, "_mmcs_commit_cuda")]
                pub(crate) fn mmcs_commit_cuda(
                    leaves: *const Matrix,
                    number_of_inputs: u32,
                    digests: *mut $field,
                    hasher: *const c_void,
                    compression: *const c_void,
                    config: &TreeBuilderConfig,
                ) -> CudaError;
            }
        }

        struct $mmcs;

        impl<Compression, Hasher> FieldMmcs<$field, Compression, Hasher> for $mmcs
        where
            Compression: SpongeHash<$field, $field>,
            Hasher: SpongeHash<$field, $field>,
        {
            fn mmcs_commit(
                leaves: Vec<Matrix>,
                digests: &mut HostSlice<$field>,
                hasher: &Hasher,
                compression: &Compression,
                config: &TreeBuilderConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::mmcs_commit_cuda(
                        leaves
                            .as_slice()
                            .as_ptr(),
                        leaves.len() as u32,
                        digests.as_mut_ptr(),
                        compression.get_handle(),
                        hasher.get_handle(),
                        config,
                    )
                    .wrap()
                }
            }
        }
    };
}
