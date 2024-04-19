extern "C" cudaError_t ${FIELD}CreateOptimizedPoseidonConstants(
  int arity,
  int full_rounds_half,
  int partial_rounds,
  const ${FIELD}::scalar_t* constants,
  device_context::DeviceContext& ctx,
  poseidon::PoseidonConstants<${FIELD}::scalar_t>* poseidon_constants);

extern "C" cudaError_t ${FIELD}InitOptimizedPoseidonConstants(
  int arity, device_context::DeviceContext& ctx, poseidon::PoseidonConstants<${FIELD}::scalar_t>* constants);

extern "C" cudaError_t ${FIELD}PoseidonHash(
  ${FIELD}::scalar_t* input,
  ${FIELD}::scalar_t* output,
  int number_of_states,
  int arity,
  const poseidon::PoseidonConstants<${FIELD}::scalar_t>& constants,
  poseidon::PoseidonConfig& config);

extern "C" cudaError_t ${FIELD}BuildPoseidonMerkleTree(
  const ${FIELD}::scalar_t* leaves,
  ${FIELD}::scalar_t* digests,
  uint32_t height,
  int arity,
  poseidon::PoseidonConstants<${FIELD}::scalar_t>& constants,
  merkle::TreeBuilderConfig& config);