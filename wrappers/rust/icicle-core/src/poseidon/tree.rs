#[macro_export]
macro_rules! impl_poseidon_tree_builder {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use super::{$field, $field_config, CudaError, DeviceContext, PoseidonHandle, TreeBuilderConfig};

            extern "C" {
                #[link_name = concat!($field_prefix, "_build_poseidon_merkle_tree")]
                pub(crate) fn build_merkle_tree(
                    leaves: *const $field,
                    digests: *mut $field,
                    height: u32,
                    input_block_len: u32,
                    poseidon_compression: PoseidonHandle,
                    poseidon_sponge: PoseidonHandle,
                    config: &TreeBuilderConfig,
                ) -> CudaError;
            }
        }

        struct PoseidonTreeBuilder;

        impl TreeBuilder<Poseidon<$field>, Poseidon<$field>, $field, $field> for PoseidonTreeBuilder {
            fn build_merkle_tree(
                leaves: &(impl HostOrDeviceSlice<$field> + ?Sized),
                digests: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                height: usize,
                input_block_len: usize,
                compression: &Poseidon<$field>,
                sponge: &Poseidon<$field>,
                config: &TreeBuilderConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::build_merkle_tree(
                        leaves.as_ptr(),
                        digests.as_mut_ptr(),
                        height as u32,
                        input_block_len as u32,
                        compression.handle,
                        sponge.handle,
                        config,
                    )
                    .wrap()
                }
            }
        }
    };
}
