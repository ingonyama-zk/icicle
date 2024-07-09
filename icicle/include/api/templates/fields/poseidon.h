extern "C" cudaError_t ${FIELD}_poseidon_create_cuda(
  poseidon::Poseidon<${FIELD}::scalar_t>** poseidon,
  unsigned int arity,
  unsigned int alpha,
  unsigned int partial_rounds,
  unsigned int full_rounds_half,
  const ${FIELD}::scalar_t* round_constants,
  const ${FIELD}::scalar_t* mds_matrix,
  const ${FIELD}::scalar_t* non_sparse_matrix,
  const ${FIELD}::scalar_t* sparse_matrices,
  const ${FIELD}::scalar_t domain_tag,
  device_context::DeviceContext& ctx);

extern "C" cudaError_t ${FIELD}_poseidon_load_cuda(
  poseidon::Poseidon<${FIELD}::scalar_t>** poseidon,
  unsigned int arity,
  device_context::DeviceContext& ctx);

extern "C" cudaError_t ${FIELD}_poseidon_hash_many_cuda(
  const poseidon::Poseidon<${FIELD}::scalar_t>* poseidon,
  const ${FIELD}::scalar_t* inputs,
  ${FIELD}::scalar_t* output,
  unsigned int number_of_states,
  unsigned int input_block_len,
  unsigned int output_len,
  hash::SpongeConfig& cfg);

extern "C" cudaError_t
  ${FIELD}_poseidon_delete_cuda(poseidon::Poseidon<${FIELD}::scalar_t>* poseidon);