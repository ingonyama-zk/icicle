#include "icicle/rings/random_sampling.h"
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
  const size_t element_size = field_t::TLC * 4;
  const size_t elements_per_hash = std::max(keccak512.output_size() / element_size, size_t(1));
  // To support elements that are larger than 32 bytes
  const size_t hashes_per_element = std::max(element_size / keccak512.output_size(), size_t(1));
  const size_t size_per_task =
    (size + RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS - 1) / RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS;

  for (uint32_t b = 0; b < cfg.batch_size; ++b) {
    field_t* batch_output = output + b * size;
    HashConfig hash_cfg{};
    tf::Taskflow taskflow;
    tf::Executor executor(RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS);
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
  const size_t element_size = field_t::TLC * 4;
  const size_t elements_per_hash = std::max(keccak512.output_size() / element_size, size_t(1));
  // To support elements that are larger than 32 bytes
  const size_t hashes_per_element = std::max(element_size / keccak512.output_size(), size_t(1));
  const size_t hashes_per_batch = size / elements_per_hash;

  const int nof_workers = std::min((int)(hashes_per_batch), get_nof_workers(cfg));
  const size_t hashes_per_worker = (hashes_per_batch + nof_workers - 1) / nof_workers;

  for (uint32_t b = 0; b < cfg.batch_size; ++b) {
    field_t* batch_output = output + b * size;
    tf::Taskflow taskflow;
    tf::Executor executor(nof_workers);
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

eIcicleError cpu_random_sampling_rq(
  const Device& device,
  uint64_t size,
  bool fast_mode,
  const std::byte* seed,
  uint64_t seed_len,
  const VecOpsConfig& config,
  Rq* output)
{
  cpu_random_sampling(device, size * Rq::d, fast_mode, seed, seed_len, config, reinterpret_cast<field_t*>(output));
  return eIcicleError::SUCCESS;
}

REGISTER_RING_ZQ_RANDOM_SAMPLING_BACKEND("CPU", cpu_random_sampling);
REGISTER_RING_RQ_RANDOM_SAMPLING_BACKEND("CPU", cpu_random_sampling_rq);