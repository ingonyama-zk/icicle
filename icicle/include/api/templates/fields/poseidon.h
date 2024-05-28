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

extern "C" cudaError_t ${FIELD}_poseidon_permute_many_cuda(
  const poseidon::Poseidon<${FIELD}::scalar_t>* poseidon,
  const ${FIELD}::scalar_t* states,
  ${FIELD}::scalar_t* output,
  unsigned int number_of_states,
  device_context::DeviceContext& ctx,
  bool is_async);

extern "C" cudaError_t ${FIELD}_poseidon_compress_many_cuda(
  const poseidon::Poseidon<${FIELD}::scalar_t>* poseidon,
  const ${FIELD}::scalar_t* states,
  ${FIELD}::scalar_t* output,
  unsigned int number_of_states,
  unsigned int offset,
  ${FIELD}::scalar_t* perm_output,
  device_context::DeviceContext& ctx,
  bool is_async);

extern "C" cudaError_t
${FIELD}_poseidon_delete_cuda(Poseidon<${FIELD}::scalar_t>* poseidon, device_context::DeviceContext& ctx);