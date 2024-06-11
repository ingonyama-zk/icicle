extern "C" cudaError_t ${FIELD}_poseidon2_create_cuda(
  poseidon2::Poseidon2<${FIELD}::scalar_t>** poseidon,
  unsigned int width,
  unsigned int alpha,
  unsigned int internal_rounds,
  unsigned int external_rounds,
  const ${FIELD}::scalar_t* round_constants,
  const ${FIELD}::scalar_t* internal_matrix_diag,
  poseidon2::MdsType mds_type,
  poseidon2::DiffusionStrategy diffusion,
  device_context::DeviceContext& ctx
);

extern "C" cudaError_t ${FIELD}_poseidon2_load_cuda(
  poseidon2::Poseidon2<${FIELD}::scalar_t>** poseidon,
  unsigned int width,
  poseidon2::MdsType mds_type,
  poseidon2::DiffusionStrategy diffusion,
  device_context::DeviceContext& ctx
);

extern "C" cudaError_t ${FIELD}_poseidon2_absorb_many_cuda(
  const poseidon2::Poseidon2<${FIELD}::scalar_t>* poseidon,
  const ${FIELD}::scalar_t* inputs,
  ${FIELD}::scalar_t* states,
  unsigned int number_of_states,
  unsigned int input_block_len,
  hash::SpongeConfig& cfg);

extern "C" cudaError_t ${FIELD}_poseidon2_squeeze_many_cuda(
  const poseidon2::Poseidon2<${FIELD}::scalar_t>* poseidon,
  const ${FIELD}::scalar_t* states,
  ${FIELD}::scalar_t* output,
  unsigned int number_of_states,
  unsigned int output_len,
  hash::SpongeConfig& cfg);

extern "C" cudaError_t ${FIELD}_poseidon2_hash_many_cuda(
  const poseidon2::Poseidon2<${FIELD}::scalar_t>* poseidon,
  const ${FIELD}::scalar_t* inputs,
  ${FIELD}::scalar_t* output,
  unsigned int number_of_states,
  unsigned int input_block_len,
  unsigned int output_len,
  hash::SpongeConfig& cfg);

extern "C" cudaError_t
  ${FIELD}_poseidon2_delete_cuda(poseidon2::Poseidon2<${FIELD}::scalar_t>* poseidon, device_context::DeviceContext& ctx);