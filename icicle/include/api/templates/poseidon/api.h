#pragma once

#ifndef ${FIELD_UPPER}_POSEIDON_API_H
#define ${FIELD_UPPER}_POSEIDON_API_H

#include "${FIELD_HEADER}"
#include <cuda_runtime.h>
#include "poseidon/poseidon.cuh"
#include "poseidon/tree/merkle.cuh"
#include "gpu-utils/device_context.cuh"

extern "C" cudaError_t ${FIELD}::PoseidonHash(
  ${FIELD}::scalar_t* input,
  ${FIELD}::scalar_t* output,
  int number_of_states,
  int arity,
  const poseidon::PoseidonConstants<${FIELD}::scalar_t>& constants,
  poseidon::PoseidonConfig& config);

extern "C" cudaError_t ${FIELD}::BuildPoseidonMerkleTree(
  const ${FIELD}::scalar_t* leaves,
  ${FIELD}::scalar_t* digests,
  uint32_t height,
  int arity,
  poseidon::PoseidonConstants<${FIELD}::scalar_t>& constants,
  merkle::TreeBuilderConfig& config);

#endif