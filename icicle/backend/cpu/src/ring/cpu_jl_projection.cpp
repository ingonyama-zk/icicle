#include "icicle/jl_projection.h"
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/hash/keccak.h"

static eIcicleError cpu_jl_projection(
  const Device& device,
  const field_t* input,
  size_t input_size,
  const std::byte* seed,
  size_t seed_len,
  const VecOpsConfig& cfg,
  field_t* output,
  size_t output_size)
{
  const size_t PI_nof_rows = output_size;
  const size_t PI_nof_cols = input_size;

  // loop rows (TODO Yuval, parallel rows with taskflow)
  // TODO Yuval: move this hashing logic out to share with the other projection API
  auto keccak512 = Keccak512::create();
  constexpr uint32_t nof_bits_per_random_element = 2; // values are {-1,0,1}
  const uint64_t nof_elements_per_hash = keccak512.output_size()
                                         << 2; // This is *8 (bytes-to-bits) and /2 (2b per elment)
  const uint32_t hashes_per_row = (PI_nof_cols + nof_elements_per_hash - 1) / nof_elements_per_hash;

  uint32_t counter = 0;
  std::vector<std::byte> hash_input(seed_len + sizeof(counter)); // 'seed||counter'
  std::memcpy(hash_input.data(), seed, seed_len);
  std::vector<std::byte> hash_output(keccak512.output_size());

  HashConfig hash_cfg{};
  for (uint32_t row_idx = 0; row_idx < PI_nof_rows; ++row_idx) {
    field_t output_value = field_t::zero();
    // iterate over the row, generate the elements and compute a partial sum
    for (uint32_t hash_idx_in_row = 0; hash_idx_in_row < hashes_per_row; ++hash_idx_in_row) {
      counter = hashes_per_row * row_idx + hash_idx_in_row;
      // copy counter value into the hash_input
      std::memcpy(hash_input.data() + seed_len, &counter, sizeof(counter));
      keccak512.hash(hash_input.data(), hash_input.size(), hash_cfg, hash_output.data());
      // inner loop: add/subtract input elements to/from output value
      for (uint32_t inner_loop_idx = 0; inner_loop_idx < nof_elements_per_hash; ++inner_loop_idx) {
        const size_t byte_idx = inner_loop_idx >> 2;          // Selects byte every 4 iterations
        const size_t bit_offset = (inner_loop_idx & 0x3) * 2; // 0, 2, 4, 6
        const std::byte raw_byte = hash_output[byte_idx];
        const uint8_t random_element_2b = (std::to_integer<uint8_t>(raw_byte) >> bit_offset) & 0x3;
        /// mapping:  00 →  0, 01 →  1, 10 → -1, 11 →  0
        const auto input_element_idx = hash_idx_in_row * nof_elements_per_hash + inner_loop_idx;
        if (input_element_idx >= input_size) { break; }
        if (random_element_2b == 0x1) { // → +1
          output_value = output_value + *(input + input_element_idx);
        } else if (random_element_2b == 0x2) { // → -1
          output_value = output_value - *(input + input_element_idx);
        }
        // 0x0 and 0x3 map to 0 → skipped
      }
    }
    output[row_idx] = output_value;
  }

  return eIcicleError::SUCCESS;
}

REGISTER_JL_PROJECTION_BACKEND("CPU", cpu_jl_projection);