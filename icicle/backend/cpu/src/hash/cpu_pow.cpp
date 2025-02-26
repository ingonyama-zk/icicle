#include "icicle/backend/hash/pow_backend.h"
#include "icicle/utils/modifiers.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace icicle {
  namespace {
    void build_chunks(
      const uint8_t* input, // Input challenge of size 32 bytes
      uint8_t* output,      // Output array to store all chunks
      uint64_t num_chunks,  // Number of chunks to generate
      uint32_t challenge_size,
      uint32_t padding_size,
      uint32_t full_size)
    {
      for (uint64_t idx = 0; idx < num_chunks; ++idx) {
        // Calculate the start position for this chunk in the output array
        uint8_t* chunk_start = output + idx * full_size;
        // Copy input_challenge
        for (int i = 0; i < challenge_size; ++i) {
          chunk_start[i] = input[i];
        }

        // Fill the padding with zeros
        for (int i = 0; i < padding_size; ++i) {
          chunk_start[challenge_size + 8 + i] = 0;
        }
      }
    }

    void
    update_chunks(uint8_t* challenge, uint64_t num_chunks, uint64_t offset, uint32_t challenge_size, uint32_t full_size)
    {
      for (uint64_t idx = 0; idx < num_chunks; ++idx) {
        uint64_t res = idx + offset;
        memcpy(challenge + idx * full_size + challenge_size, &res, sizeof(uint64_t));
      }
    }

    void find_solving(
      uint8_t* hashes,
      uint32_t hash_size,
      uint32_t length,
      uint64_t threshold,
      uint64_t offset,
      bool& found,
      uint64_t& nonce,
      uint64_t& mined_hash)
    {
      for (uint64_t idx = 0; idx < length; ++idx) {
        uint64_t candidate = *(uint64_t*)(hashes + hash_size * idx);
        if (candidate < threshold) {
          found = true;
          nonce = idx + offset;
          mined_hash = candidate;
          break;
        }
      }
    }
  } // namespace

  eIcicleError cpu_pow(
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint32_t hash_size,
    uint8_t solution_bits,
    const PowConfig& config,
    bool& found,
    uint64_t& nonce,
    uint64_t& mined_hash)
  {
    if (solution_bits < 1 || solution_bits > 60) {
      ICICLE_LOG_ERROR << "invalid solution_bits value";
      return eIcicleError::INVALID_ARGUMENT;
    }

    uint64_t threshold = (uint64_t)1 << (64 - solution_bits);
    uint32_t full_size = challenge_size + sizeof(uint64_t) + config.padding_size;

    uint64_t grid_size = 1024;
    uint64_t max_iterations =
      ((uint64_t)(-1) - grid_size + 1) /
      grid_size; // max 2^64 - 1, number of kernel calls ceil((2^64 - 1) / (num_blocks * num_threads))

    // setup input
    uint8_t* inputs = new uint8_t[full_size * grid_size];
    build_chunks(
      reinterpret_cast<const uint8_t*>(challenge), inputs, grid_size, challenge_size, config.padding_size, full_size);

    uint64_t i = 0;
    uint64_t offset = 0;

    uint8_t* outputs = new uint8_t[hash_size * grid_size];

    // hash config
    auto cfg = default_hash_config();
    cfg.batch = grid_size;
    cfg.stream = config.stream;
    cfg.is_async = config.is_async;

    // main solving loop
    do {
      // set new nonces in the input
      update_chunks(inputs, grid_size, offset, challenge_size, full_size);
      // batch hash
      hasher.hash(inputs, full_size, cfg, outputs);
      // check if the value less than the threshold exists in computed hashes
      find_solving(outputs, hash_size, grid_size, threshold, offset, found, nonce, mined_hash);
      ++i;
      offset += grid_size;
    } while ((!found) && (i < max_iterations)); // while value not found or checked all possible nonces

    delete[] inputs;
    delete[] outputs;

    return eIcicleError::SUCCESS;
  }

  eIcicleError cpu_pow_verify(
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint32_t hash_size,
    uint8_t solution_bits,
    const PowConfig& config,
    uint64_t nonce,
    bool& is_correct,
    uint64_t& mined_hash)
  {
    if (solution_bits < 1 || solution_bits > 60) {
      ICICLE_LOG_ERROR << "invalid solution_bits value";
      return eIcicleError::INVALID_ARGUMENT;
    }

    uint64_t threshold = (uint64_t)1 << (64 - solution_bits);
    uint32_t full_size = challenge_size + sizeof(uint64_t) + config.padding_size;

    std::byte* input = new std::byte[full_size];

    std::copy(challenge, challenge + challenge_size, input);
    uint64_t* input_nonce = (uint64_t*)(input + challenge_size);
    *input_nonce = nonce;

    std::memset(input + challenge_size + 8, 0, config.padding_size);

    uint8_t* output = new uint8_t[hash_size];

    // hash config
    auto cfg = default_hash_config();
    cfg.stream = config.stream;
    cfg.is_async = config.is_async;

    hasher.hash(input, full_size, cfg, output);

    mined_hash = *(uint64_t*)output;
    is_correct = mined_hash < threshold;

    delete[] input;
    delete[] output;

    return eIcicleError::SUCCESS;
  }

  eIcicleError pow_solver_cpu_backend(
    const Device& device,
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    bool& found,
    uint64_t& nonce,
    uint64_t& mined_hash)
  {
    auto err =
      cpu_pow(hasher, challenge, challenge_size, hasher.output_size(), solution_bits, config, found, nonce, mined_hash);
    return err;
  }

  eIcicleError pow_verify_cpu_backend(
    const Device& device,
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    uint64_t nonce,
    bool& is_correct,
    uint64_t& mined_hash)
  {
    auto err = cpu_pow_verify(
      hasher, challenge, challenge_size, hasher.output_size(), solution_bits, config, nonce, is_correct, mined_hash);
    return err;
  }

  REGISTER_POW_SOLVER_BACKEND("CPU", pow_solver_cpu_backend);
  REGISTER_POW_VERIFY_BACKEND("CPU", pow_verify_cpu_backend);
} // namespace icicle