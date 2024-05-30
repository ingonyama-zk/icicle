#[macro_export]
macro_rules! impl_poseidon2_tree_builder {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use super::{$field, $field_config, CudaError, DeviceContext, Poseidon2Handle, TreeBuilderConfig};

            extern "C" {
                #[link_name = concat!($field_prefix, "_build_poseidon2_merkle_tree")]
                pub(crate) fn build_merkle_tree(
                    leaves: *const $field,
                    digests: *mut $field,
                    height: u32,
                    input_block_len: u32,
                    poseidon_compression: Poseidon2Handle,
                    poseidon_sponge: Poseidon2Handle,
                    config: &TreeBuilderConfig,
                ) -> CudaError;
            }
        }

        struct Poseidon2TreeBuilder;

        impl TreeBuilder<Poseidon2<$field>, Poseidon2<$field>, $field, $field> for Poseidon2TreeBuilder {
            fn build_merkle_tree(
                leaves: &(impl HostOrDeviceSlice<$field> + ?Sized),
                digests: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                height: usize,
                input_block_len: usize,
                compression: &Poseidon2<$field>,
                sponge: &Poseidon2<$field>,
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
