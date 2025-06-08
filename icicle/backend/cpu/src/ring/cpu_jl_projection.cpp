#include "icicle/jl_projection.h"
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/hash/keccak.h"
#include "taskflow/taskflow.hpp"

/// @brief CPU backend implementation of the JL projection algorithm.
/// Projects an input vector of size N (`input_size`) into a lower-dimensional
/// output vector of size 256 (or `output_size`) using a sparse random matrix.
/// The random matrix is implicitly generated on-the-fly from a seed and counter,
/// with entries in {-1, 0, 1} sampled via 2-bit decoding from a Keccak512 hash.
///
/// Matrix entries are generated with the following 2-bit encoding:
///     00 →  0
///     01 → +1
///     10 → -1
///     11 →  0
///

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

static eIcicleError cpu_jl_projection(
  const Device& device,
  const field_t* input,
  size_t input_size,
  const std::byte* seed,
  size_t seed_len,
  const VecOpsConfig& config,
  field_t* output,
  size_t output_size)
{
  if (!input || !output || !seed) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer.";
    return eIcicleError::INVALID_POINTER;
  }

  if (input_size == 0 || output_size == 0 || seed_len == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: zero size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.batch_size != 1) {
    ICICLE_LOG_ERROR << "Unsupported config: JL projection does not support batch.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const size_t num_rows = output_size; // Number of output values
  const size_t num_cols = input_size;  // Length of input vector

  auto keccak512 = Keccak512::create();
  constexpr uint32_t bits_per_entry = 2;                                        // 2 bits per matrix entry
  const size_t entries_per_hash = keccak512.output_size() * 8 / bits_per_entry; // 4 × hash bytes
  const size_t hashes_per_row = (num_cols + entries_per_hash - 1) / entries_per_hash;

  const int nof_workers = get_nof_workers(config);
  tf::Taskflow taskflow;
  tf::Executor executor(nof_workers);

  for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
    taskflow.emplace([=]() {
      // Allocate buffers for hash input (seed || counter) and hash output
      std::vector<std::byte> hash_input(seed_len + sizeof(uint32_t));
      std::memcpy(hash_input.data(), seed, seed_len);
      std::vector<std::byte> hash_output(keccak512.output_size());

      HashConfig hash_cfg{}; // Default config for Keccak

      field_t acc = field_t::zero(); // Accumulator for this row

      // Loop over all hash blocks needed to cover the full input
      for (size_t hash_idx = 0; hash_idx < hashes_per_row; ++hash_idx) {
        uint32_t counter = static_cast<uint32_t>(row_idx * hashes_per_row + hash_idx);

        // Concatenate seed || counter
        std::memcpy(hash_input.data() + seed_len, &counter, sizeof(counter));

        // Compute Keccak512(seed || counter) → hash_output
        keccak512.hash(hash_input.data(), hash_input.size(), hash_cfg, hash_output.data());

        // Each hash output encodes up to 256 2-bit matrix entries
        for (size_t entry_idx = 0; entry_idx < entries_per_hash; ++entry_idx) {
          const size_t input_idx = hash_idx * entries_per_hash + entry_idx;
          if (input_idx >= input_size) break;

          // Extract 2 bits for the current matrix entry
          const size_t byte_idx = entry_idx >> 2;          // 4 entries per byte
          const size_t bit_offset = (entry_idx & 0x3) * 2; // shift = 0, 2, 4, 6
          const uint8_t byte = std::to_integer<uint8_t>(hash_output[byte_idx]);
          const uint8_t rnd_2b = (byte >> bit_offset) & 0x3;

          // Map 2-bit pattern to matrix value: +1, -1, or 0
          if (rnd_2b == 0x1) { // 01 → +1
            acc = acc + input[input_idx];
          } else if (rnd_2b == 0x2) { // 10 → -1
            acc = acc - input[input_idx];
          }
          // 00 and 11 → zero, skip
        }
      }
      output[row_idx] = acc;
    });
  }

  executor.run(taskflow).wait();
  taskflow.clear();

  return eIcicleError::SUCCESS;
}

static eIcicleError cpu_get_jl_matrix_rows(
  const Device& device,
  const std::byte* seed,
  size_t seed_len,
  size_t row_size,
  size_t start_row,
  size_t num_rows,
  const VecOpsConfig& cfg,
  field_t* output)
{
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_JL_PROJECTION_BACKEND("CPU", cpu_jl_projection, cpu_get_jl_matrix_rows);