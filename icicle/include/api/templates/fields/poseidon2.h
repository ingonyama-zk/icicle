extern "C" cudaError_t ${FIELD}_create_poseidon2_constants_cuda(
  unsigned int width,
  unsigned int alpha,
  unsigned int internal_rounds,
  unsigned int external_rounds,
  const ${FIELD}::scalar_t* round_constants,
  const ${FIELD}::scalar_t* internal_matrix_diag,
  poseidon2::MdsType mds_type,
  poseidon2::DiffusionStrategy diffusion,
  device_context::DeviceContext& ctx,
  poseidon2::Poseidon2Constants<${FIELD}::scalar_t>* poseidon_constants);

extern "C" cudaError_t ${FIELD}_init_poseidon2_constants_cuda(
  unsigned int width,
  poseidon2::MdsType mds_type,
  poseidon2::DiffusionStrategy diffusion,
  device_context::DeviceContext& ctx,
  poseidon2::Poseidon2Constants<${FIELD}::scalar_t>* poseidon_constants);

extern "C" cudaError_t ${FIELD}_poseidon2_permute_many_cuda(
  const ${FIELD}::scalar_t* states,
  ${FIELD}::scalar_t* output,
  unsigned int number_of_states,
  const poseidon2::Poseidon2<${FIELD}::scalar_t>* poseidon,
  device_context::DeviceContext& ctx
);

extern "C" cudaError_t ${FIELD}_poseidon2_compress_many_cuda(
  const ${FIELD}::scalar_t* states,
  ${FIELD}::scalar_t* output,
  unsigned int number_of_states,
  unsigned int rate,
  const poseidon2::Poseidon2<${FIELD}::scalar_t>* poseidon,
  device_context::DeviceContext& ctx,
  unsigned int offset,
  ${FIELD}::scalar_t* perm_output
);

extern "C" cudaError_t ${FIELD}_release_poseidon2_constants_cuda(
  poseidon2::Poseidon2Constants<${FIELD}::scalar_t>* constants,
  device_context::DeviceContext& ctx);