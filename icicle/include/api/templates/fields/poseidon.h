extern "C" cudaError_t ${FIELD}_create_optimized_poseidon_constants_cuda(
  int arity,
  int full_rounds_half,
  int partial_rounds,
  const ${FIELD}::scalar_t* constants,
  device_context::DeviceContext& ctx,
  poseidon::PoseidonConstants<${FIELD}::scalar_t>* poseidon_constants);

extern "C" cudaError_t ${FIELD}_init_optimized_poseidon_constants_cuda(
  int arity, device_context::DeviceContext& ctx, poseidon::PoseidonConstants<${FIELD}::scalar_t>* constants);

extern "C" cudaError_t ${FIELD}_poseidon_hash_cuda(
  ${FIELD}::scalar_t* input,
  ${FIELD}::scalar_t* output,
  int number_of_states,
  int arity,
  const poseidon::PoseidonConstants<${FIELD}::scalar_t>& constants,
  poseidon::PoseidonConfig& config);

extern "C" cudaError_t ${FIELD}_build_poseidon_merkle_tree(
  const ${FIELD}::scalar_t* leaves,
  ${FIELD}::scalar_t* digests,
  uint32_t height,
  int arity,
  poseidon::PoseidonConstants<${FIELD}::scalar_t>& constants,
  merkle::TreeBuilderConfig& config);