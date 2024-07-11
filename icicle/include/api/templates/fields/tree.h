extern "C" cudaError_t ${FIELD}_build_merkle_tree(
  const ${FIELD}::scalar_t* leaves,
  ${FIELD}::scalar_t* digests,
  unsigned int height,
  unsigned int input_block_len, 
  const hash::SpongeHasher<${FIELD}::scalar_t, ${FIELD}::scalar_t>* compression,
  const hash::SpongeHasher<${FIELD}::scalar_t, ${FIELD}::scalar_t>* bottom_layer,
  const merkle_tree::TreeBuilderConfig& tree_config);

  extern "C" cudaError_t ${FIELD}_mmcs_commit_cuda(
    const matrix::Matrix<${FIELD}::scalar_t>* leaves,
    unsigned int number_of_inputs,
    ${FIELD}::scalar_t* digests,
    const hash::SpongeHasher<${FIELD}::scalar_t, ${FIELD}::scalar_t>* hasher,
    const hash::SpongeHasher<${FIELD}::scalar_t, ${FIELD}::scalar_t>* compression,
    const merkle_tree::TreeBuilderConfig& tree_config);