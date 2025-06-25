#include "icicle/backend/vec_ops_backend.h"
#include "icicle/hash/keccak.h"
#include "taskflow/taskflow.hpp"

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

void fast_mode_random_sampling(
  uint64_t size, const std::byte* seed, uint64_t seed_len, const VecOpsConfig& cfg, field_t* output)
{
  // Use keccak to get deterministic uniform distribution
  auto keccak512 = Keccak512::create();
  const size_t element_size = sizeof(field_t::limbs_storage);
  const size_t elements_per_hash = std::max(keccak512.output_size() / element_size, size_t(1));
  // To support elements that are larger than 32 bytes
  const size_t hashes_per_element =
    std::max((element_size + keccak512.output_size() - 1) / keccak512.output_size(), size_t(1));
  const size_t size_per_task =
    (size + RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS - 1) / RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS;

  tf::Taskflow taskflow;
  tf::Executor executor(get_nof_workers(cfg));
  for (uint32_t b = 0; b < cfg.batch_size; ++b) {
    field_t* batch_output = output + b * size;
    HashConfig hash_cfg{};
    for (uint64_t t = 0; t < RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS; ++t) {
      taskflow.emplace([=]() {
        std::vector<std::byte> hash_input(seed_len + sizeof(b) + sizeof(uint64_t));
        std::memcpy(hash_input.data(), seed, seed_len);
        std::memcpy(hash_input.data() + seed_len, &b, sizeof(b));
        std::memcpy(hash_input.data() + seed_len + sizeof(b), &t, sizeof(t));
        std::vector<uint64_t> hash_output(keccak512.output_size() / sizeof(uint64_t) * hashes_per_element);

        keccak512.hash(
          hash_input.data(), hash_input.size(), hash_cfg, reinterpret_cast<std::byte*>(hash_output.data()));
        for (int i = 1; i < hashes_per_element; i++) {
          keccak512.hash(
            reinterpret_cast<std::byte*>(hash_output.data()) + (i - 1) * keccak512.output_size(),
            keccak512.output_size(), hash_cfg,
            reinterpret_cast<std::byte*>(hash_output.data()) + i * keccak512.output_size());
        }

        field_t prev_element = field_t::reduce_from_bytes(reinterpret_cast<std::byte*>(hash_output.data()));
        batch_output[t * size_per_task] = prev_element;
        for (int i = 1; i < size_per_task && (t * size_per_task + i) < size; i++) {
          field_t next_element = field_t::sqr(prev_element);
          prev_element = next_element;
          batch_output[t * size_per_task + i] = next_element;
        }
      });
    }
    executor.run(taskflow).wait();
    taskflow.clear();
  }
}

void slow_mode_random_sampling(
  uint64_t size, const std::byte* seed, uint64_t seed_len, const VecOpsConfig& cfg, field_t* output)
{
  // Use keccak to get deterministic uniform distribution
  auto keccak512 = Keccak512::create();
  const size_t element_size = sizeof(field_t::limbs_storage);
  const size_t elements_per_hash = std::max(keccak512.output_size() / element_size, size_t(1));
  // To support elements that are larger than 32 bytes
  const size_t hashes_per_element = std::max(element_size / keccak512.output_size(), size_t(1));
  const size_t hashes_per_batch = size / elements_per_hash;

  const int nof_workers = std::min((int)(hashes_per_batch), get_nof_workers(cfg));
  const size_t hashes_per_worker = (hashes_per_batch + nof_workers - 1) / nof_workers;

  tf::Taskflow taskflow;
  tf::Executor executor(nof_workers);
  for (uint32_t b = 0; b < cfg.batch_size; ++b) {
    field_t* batch_output = output + b * size;
    for (uint32_t w = 0; w < nof_workers; ++w) {
      taskflow.emplace([=]() {
        HashConfig hash_cfg{};
        std::vector<std::byte> hash_input(seed_len + sizeof(b) + sizeof(size));
        std::memcpy(hash_input.data(), seed, seed_len);
        std::memcpy(hash_input.data() + seed_len, &b, sizeof(b));
        std::vector<uint64_t> hash_output(keccak512.output_size() / sizeof(uint64_t) * hashes_per_element);
        for (size_t counter = w * hashes_per_worker;
             counter < (w + 1) * hashes_per_worker && counter < hashes_per_batch; counter++) {
          std::memcpy(hash_input.data() + seed_len + sizeof(b), &counter, sizeof(counter));

          keccak512.hash(
            hash_input.data(), hash_input.size(), hash_cfg, reinterpret_cast<std::byte*>(hash_output.data()));
          for (int i = 1; i < hashes_per_element; i++) {
            keccak512.hash(
              reinterpret_cast<std::byte*>(hash_output.data()) + (i - 1) * keccak512.output_size(),
              keccak512.output_size(), hash_cfg,
              reinterpret_cast<std::byte*>(hash_output.data()) + i * keccak512.output_size());
          }
          for (int i = 0; i < elements_per_hash; i++) {
            batch_output[counter * elements_per_hash + i] =
              field_t::reduce_from_bytes(reinterpret_cast<std::byte*>(hash_output.data()) + i * element_size);
          }
        }
      });
    }
    executor.run(taskflow).wait();
    taskflow.clear();
  }
}

eIcicleError cpu_random_sampling(
  const Device& device,
  uint64_t size,
  bool fast_mode,
  const std::byte* seed,
  uint64_t seed_len,
  const VecOpsConfig& cfg,
  field_t* output)
{
  if (!seed || !output) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer.";
    return eIcicleError::INVALID_POINTER;
  }

  if (seed_len == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: zero seed length.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (fast_mode) {
    fast_mode_random_sampling(size, seed, seed_len, cfg, output);
  } else {
    slow_mode_random_sampling(size, seed, seed_len, cfg, output);
  }

  return eIcicleError::SUCCESS;
}

REGISTER_RING_ZQ_RANDOM_SAMPLING_BACKEND("CPU", cpu_random_sampling);

struct RandomBitIterator {
  uint64_t keccak_buffer[8];
  size_t limb_idx = 0;
  size_t bit_idx = 0;
  uint64_t lfsr_state = 0;

  // Initialize with Keccak output (must be at least 8 uint64_t)
  RandomBitIterator(const std::vector<uint64_t>& keccak_output)
  {
    for (int i = 0; i < 8; ++i)
      keccak_buffer[i] = keccak_output[i];
    limb_idx = 0;
    bit_idx = 0;
    lfsr_state = keccak_buffer[7]; // Seed LFSR from last Keccak word
  }

  inline uint64_t lfsr64(uint64_t x)
  {
    uint64_t lsb = x & 1;
    x >>= 1;
    if (lsb) x ^= 0xD800000000000000ULL;
    return x;
  }

  int next_bit()
  {
    int bit;
    if (limb_idx < 8) {
      bit = (keccak_buffer[limb_idx] >> bit_idx) & 1;
    } else {
      bit = (lfsr_state >> bit_idx) & 1;
    }
    ++bit_idx;
    if (bit_idx == 64) {
      bit_idx = 0;
      ++limb_idx;
      if (limb_idx >= 8) { lfsr_state = lfsr64(lfsr_state); }
    }
    return bit;
  }
};

// Cross shuffles two adjacent ranges of an array as described in the paper
// https://arxiv.org/pdf/1508.03167
template <typename T>
void merge_shuffle(
  T* array, uint32_t size_a, uint32_t size_b, uint32_t index_bits, RandomBitIterator& random_bit_iterator)
{
  int i = 0;
  int j = size_a;
  int n = size_a + size_b;
  while (true) {
    if (random_bit_iterator.next_bit()) {
      if (j == n) { break; }
      std::swap(array[i], array[j]);
      ++j;
    } else {
      if (i == j) { break; }
      ++i;
    }
  }
  for (i = i ? i : 1; i < n; i++) {
    uint32_t m = 0;
    for (int b = 0; b < index_bits; b++) {
      m |= random_bit_iterator.next_bit();
      m <<= 1;
    }
    m = m % i;
    std::swap(array[i], array[m]);
  }
}

eIcicleError cpu_challenge_space_polynomials_sampling(
  const Device& device,
  const std::byte* seed,
  uint64_t seed_len,
  size_t size,
  uint32_t ones,
  uint32_t twos,
  const VecOpsConfig& cfg,
  Rq* output)
{
  if (!seed || !output) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer.";
    return eIcicleError::INVALID_POINTER;
  }

  if (seed_len == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: zero seed length.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (ones + twos > Rq::d) {
    ICICLE_LOG_ERROR << "Invalid argument: number of coefficients > polynomial degree.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  auto keccak512 = Keccak512::create();
  const size_t polynomials_per_hash = CHALLENGE_SPACE_POLYNOMIALS_SAMPLING_POLYNOMIALS_PER_HASH;
  const size_t total_hashes = (size + polynomials_per_hash - 1) / polynomials_per_hash;

  static const field_t two = field_t::one() + field_t::one();
  static const field_t neg_two = field_t::neg(two);
  static const field_t neg_one = field_t::neg(field_t::one());

  const size_t nof_workers = std::min((size_t)get_nof_workers(cfg), total_hashes);
  const size_t size_per_worker = (size + nof_workers - 1) / nof_workers;
  const size_t hashes_per_worker = (size_per_worker + polynomials_per_hash - 1) / polynomials_per_hash;

  tf::Taskflow taskflow;
  tf::Executor executor(nof_workers);

  for (size_t w = 0; w < nof_workers; ++w) {
    taskflow.emplace([=]() {
      for (size_t hash_idx = w * hashes_per_worker;
           hash_idx < (w + 1) * hashes_per_worker && hash_idx * polynomials_per_hash < size; hash_idx++) {
        Rq* output_polynomials = output + hash_idx * polynomials_per_hash;
        size_t compute_polynomials = std::min(polynomials_per_hash, size - hash_idx * polynomials_per_hash);

        // Setup the random bits iterator
        HashConfig hash_cfg{};
        std::vector<std::byte> hash_input(seed_len + sizeof(hash_idx));
        std::memcpy(hash_input.data(), seed, seed_len);
        std::memcpy(hash_input.data() + seed_len, &hash_idx, sizeof(hash_idx));
        std::vector<uint64_t> hash_output(keccak512.output_size());
        keccak512.hash(hash_input.data(), hash_input.size(), hash_cfg, hash_output.data());
        RandomBitIterator random_bit_iterator(hash_output);

        // Initialize polynomial with coefficients and randomly flip signs of ones and twos coefficients
        // [1, -1, ..., 1, 2, -2, ..., 2, 0, ..., 0]
        for (uint32_t i = 0; i < compute_polynomials; ++i) {
          for (uint32_t l = 0; l < ones; ++l) {
            output_polynomials[i].values[l] = random_bit_iterator.next_bit() ? field_t::one() : neg_one;
          }
          for (uint32_t m = ones; m < ones + twos; ++m) {
            output_polynomials[i].values[m] = random_bit_iterator.next_bit() ? two : neg_two;
          }
          // TODO: memset here?
          for (uint32_t k = ones + twos; k < Rq::d; ++k) {
            output_polynomials[i].values[k] = field_t::zero();
          }
        }

        for (uint32_t i = 0; i < compute_polynomials; ++i) {
          // Do merge shuffle of 1s and 2s
          merge_shuffle(
            output_polynomials[i].values, ones, twos, std::ceil(std::log2(ones + twos)), random_bit_iterator);
          // Do merge shuffle of shuffled 1s and 2s and zeroes
          merge_shuffle(
            output_polynomials[i].values, ones + twos, Rq::d - ones - twos, std::ceil(std::log2(Rq::d)),
            random_bit_iterator);
        }
      }
    });
  }

  executor.run(taskflow).wait();
  taskflow.clear();

  return eIcicleError::SUCCESS;
}

REGISTER_CHALLENGE_SPACE_POLYNOMIALS_SAMPLING_BACKEND("CPU", cpu_challenge_space_polynomials_sampling);
