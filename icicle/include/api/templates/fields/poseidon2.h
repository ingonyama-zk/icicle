extern "C" cudaError_t ${FIELD}_create_poseidon2_constants_cuda(
  int width,
  int alpha,
  int internal_rounds,
  int external_rounds,
  const ${FIELD}::scalar_t* round_constants,
  const ${FIELD}::scalar_t* internal_matrix_diag,
  poseidon2::MdsType mds_type,
  poseidon2::DiffusionStrategy diffusion,
  device_context::DeviceContext& ctx,
  poseidon2::Poseidon2Constants<${FIELD}::scalar_t>* poseidon_constants);

extern "C" cudaError_t ${FIELD}_init_poseidon2_constants_cuda(
  int width,
  poseidon2::MdsType mds_type,
  poseidon2::DiffusionStrategy diffusion,
  device_context::DeviceContext& ctx,
  poseidon2::Poseidon2Constants<${FIELD}::scalar_t>* poseidon_constants);

extern "C" cudaError_t ${FIELD}_poseidon2_hash_cuda(
  const ${FIELD}::scalar_t* input,
  ${FIELD}::scalar_t* output,
  int number_of_states,
  int width,
  const poseidon2::Poseidon2Constants<${FIELD}::scalar_t>& constants,
  poseidon2::Poseidon2Config& config);

extern "C" cudaError_t ${FIELD}_release_poseidon2_constants_cuda(
  poseidon2::Poseidon2Constants<${FIELD}::scalar_t>* constants,
  device_context::DeviceContext& ctx);