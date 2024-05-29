extern "C" cudaError_t ${FIELD}_build_poseidon_merkle_tree(
  const ${FIELD}::scalar_t* leaves,
  ${FIELD}::scalar_t* digests,
  unsigned int height,
  unsigned int arity,
  unsigned int input_block_len, 
  const poseidon::Poseidon<${FIELD}::scalar_t>* poseidon_compression,
  const poseidon::Poseidon<${FIELD}::scalar_t>* poseidon_sponge,
  const hash::SpongeConfig& sponge_config,
  const merkle_tree::TreeBuilderConfig& tree_config);