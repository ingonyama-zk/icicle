#pragma once
// #include "icicle/backend/ntt_backend.h"
// #include "icicle/errors.h"
// #include "icicle/runtime.h"
// #include "icicle/utils/log.h"
// #include "icicle/fields/field_config.h"
// #include "icicle/vec_ops.h"
#include "icicle/utils/log.h"
#include "ntt_tasks.h"
#include "ntt_tasks_ref.h"
#include <iostream>

// #include <thread>
// #include <vector>
// #include <chrono>
// #include <algorithm>
// #include <iostream>
// #include <cmath>
// #include <cstdint>
// #include <memory>
// #include <mutex>

using namespace field_config;
using namespace icicle;
#define PARALLEL 0

namespace ntt_cpu {
  // main
  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError
  cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir direction, const NTTConfig<S>& config, E* output)
  {
    // auto start_hierarchy1_push_tasks = std::chrono::high_resolution_clock::now();
    // auto start_handle_pushed_tasks = std::chrono::high_resolution_clock::now();
    // auto end_hierarchy1_push_tasks = std::chrono::high_resolution_clock::now();
    // auto end_handle_pushed_tasks = std::chrono::high_resolution_clock::now();

    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
    if (size & (size - 1)) {
      ICICLE_LOG_ERROR << "Size must be a power of 2. size = " << size;
      return eIcicleError::INVALID_ARGUMENT;
    }
    if (size > domain_max_size) {
      ICICLE_LOG_ERROR << "Size is too large for domain. size = " << size << ", domain_max_size = " << domain_max_size;
      return eIcicleError::INVALID_ARGUMENT;
    }

    const int logn = int(log2(size));
    const uint64_t total_memory_size = size * config.batch_size;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    NttCpu<S, E> ntt(logn, direction, config, domain_max_size, twiddles);
    NttTaskCordinates ntt_task_cordinates = {0, 0, 0, 0, 0};
    NttTasksManager<S, E> ntt_tasks_manager(logn);
    const int nof_threads = std::thread::hardware_concurrency();
    auto tasks_manager = new TasksManager<NttTask<S, E>>(nof_threads - 1);
    // auto tasks_manager = new TasksManager<NttTask<S, E>>(1);
    NttTask<S, E>* task_slot;
    std::unique_ptr<S[]> arbitrary_coset = nullptr;
    const int coset_stride = ntt.find_or_generate_coset(arbitrary_coset);

    ntt.copy_and_reorder_if_needed(input, output);
    if (config.coset_gen != S::one() && direction == NTTDir::kForward) {
      ntt.coset_mul(output, twiddles, coset_stride, arbitrary_coset);
    }

    if (logn > H1) {
      int sunbtt_plus_batch_logn = ntt.ntt_sub_logn.h1_layers_sub_logn[0] + int(log2(config.batch_size));
      int log_nof_h1_subntts_todo_in_parallel = sunbtt_plus_batch_logn < H1 ? H1 - sunbtt_plus_batch_logn : 0;
      int nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      int log_nof_subntts_chunks = ntt.ntt_sub_logn.h1_layers_sub_logn[1] - log_nof_h1_subntts_todo_in_parallel;
      int nof_subntts_chunks = 1 << log_nof_subntts_chunks;

      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel;
             h1_subntt_idx_in_chunck++) {
          ntt_task_cordinates.h1_subntt_idx =
            h1_subntts_chunck_idx * nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;
          ntt.hierarchy1_push_tasks(output, ntt_task_cordinates, ntt_tasks_manager);
        }
        ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 0);
      }

      ntt.h1_reorder(output);
      ntt_task_cordinates.h1_layer_idx = 1;
      sunbtt_plus_batch_logn = ntt.ntt_sub_logn.h1_layers_sub_logn[1] + int(log2(config.batch_size));
      log_nof_h1_subntts_todo_in_parallel = sunbtt_plus_batch_logn < H1 ? H1 - sunbtt_plus_batch_logn : 0;
      nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      log_nof_subntts_chunks = ntt.ntt_sub_logn.h1_layers_sub_logn[0] - log_nof_h1_subntts_todo_in_parallel;
      nof_subntts_chunks = 1 << log_nof_subntts_chunks;

      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel;
             h1_subntt_idx_in_chunck++) {
          ntt_task_cordinates.h1_subntt_idx =
            h1_subntts_chunck_idx * nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;
          ntt.hierarchy1_push_tasks(output, ntt_task_cordinates, ntt_tasks_manager);
        }
        ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 1);
      }

      ntt_task_cordinates.h1_subntt_idx =
        0; // reset so that reorder_and_refactor_if_needed will calculate the correct memory index
      if (config.columns_batch) {
        ntt.reorder_and_refactor_if_needed(output, ntt_task_cordinates, true);
      } else {
        for (int b = 0; b < config.batch_size; b++) {
          ntt.reorder_and_refactor_if_needed(output + b * size, ntt_task_cordinates, true);
        }
      }
    } else {
      // start_hierarchy1_push_tasks = std::chrono::high_resolution_clock::now();
      ntt.hierarchy1_push_tasks(output, ntt_task_cordinates, ntt_tasks_manager);
      // end_hierarchy1_push_tasks = std::chrono::high_resolution_clock::now();

      // start_handle_pushed_tasks = std::chrono::high_resolution_clock::now();
      ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 0);
      // end_handle_pushed_tasks = std::chrono::high_resolution_clock::now();
    }

    if (direction == NTTDir::kInverse) { // TODO SHANIE - do that in parallel
      S inv_size = S::inv_log_size(logn);
      for (uint64_t i = 0; i < total_memory_size; ++i) {
        output[i] = output[i] * inv_size;
      }
      if (config.coset_gen != S::one()) { ntt.coset_mul(output, twiddles, coset_stride, arbitrary_coset); }
    }

    if (config.ordering == Ordering::kNR || config.ordering == Ordering::kRR) {
      ntt_task_cordinates = {0, 0, 0, 0, 0};
      ntt.reorder_by_bit_reverse(ntt_task_cordinates, output, true);
    }
    // auto duration_hierarchy1_push_tasks = std::chrono::duration<double, std::milli>(end_hierarchy1_push_tasks -
    // start_hierarchy1_push_tasks).count(); auto duration_handle_pushed_tasks = std::chrono::duration<double,
    // std::milli>(end_handle_pushed_tasks - start_handle_pushed_tasks).count(); ICICLE_LOG_INFO << std::fixed <<
    // std::setprecision(3)
    //                  << "Time spent in hierarchy1_push_tasks: " << duration_hierarchy1_push_tasks << " ms";
    // ICICLE_LOG_INFO << std::fixed << std::setprecision(3)
    //                  << "Time spent in handle_pushed_tasks: " << duration_handle_pushed_tasks << " ms";
    // std:std::cout << std::fixed << std::setprecision(3)
    //                   << duration_handle_pushed_tasks << " \n";
    // std::cout << std::fixed << std::setprecision(3) << ntt.duration_total/3072 << std::endl;

    delete tasks_manager;
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError cpu_ntt_ref(
    const Device& device, const E* input, uint64_t size, NTTDir direction, const NTTConfig<S>& config, E* output)
  {
    if (size & (size - 1)) {
      ICICLE_LOG_ERROR << "Size must be a power of 2. size = " << size;
      return eIcicleError::INVALID_ARGUMENT;
    }
    const int logn = int(log2(size));
    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
    if (size > domain_max_size) {
      ICICLE_LOG_ERROR << "Size is too large for domain. size = " << size << ", domain_max_size = " << domain_max_size;
      return eIcicleError::INVALID_ARGUMENT;
    }
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    NttCpuRef<S, E> ntt(logn, direction, config, domain_max_size, twiddles);
    NttTaskCordinatesRef ntt_task_cordinates = {0, 0, 0, 0, 0};
    NttTasksManagerRef<S, E> ntt_tasks_manager(logn);
    int nof_threads = std::thread::hardware_concurrency();
    auto tasks_manager = new TasksManager<NttTaskRef<S, E>>(nof_threads - 1);
    // auto tasks_manager = new TasksManager<NttTaskRef<S, E>>(1);

    int coset_stride = 0;
    std::unique_ptr<S[]> arbitrary_coset = nullptr;
    if (config.coset_gen != S::one()) {
      try {
        coset_stride =
          CpuNttDomain<S>::s_ntt_domain.coset_index.at(config.coset_gen); // Coset generator found in twiddles
      } catch (const std::out_of_range& oor) { // Coset generator not found in twiddles. Calculating arbitrary coset
        arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset[0] = S::one();
        S coset_gen =
          direction == NTTDir::kForward ? config.coset_gen : S::inverse(config.coset_gen); // inverse for INTT
        for (int i = 1; i <= domain_max_size; i++) {
          arbitrary_coset[i] = arbitrary_coset[i - 1] * coset_gen;
        }
      }
    }
    uint64_t total_memory_size = size * config.batch_size;
    std::copy(input, input + total_memory_size, output);
    if (config.ordering == Ordering::kRN || config.ordering == Ordering::kRR) {
      ntt.reorder_by_bit_reverse(
        ntt_task_cordinates, output,
        true); // TODO - check if access the fixed indexes instead of reordering may be more efficient?
    }

    if (config.coset_gen != S::one() && direction == NTTDir::kForward) {
      ntt.coset_mul(output, twiddles, coset_stride, arbitrary_coset);
    }
    NttTaskRef<S, E>* task_slot;

    if (logn > 15) {
      ntt.reorder_input(output);
      int sunbtt_plus_batch_logn = ntt.ntt_sub_logn.h1_layers_sub_logn[0] + int(log2(config.batch_size));
      int log_nof_h1_subntts_todo_in_parallel = sunbtt_plus_batch_logn < 15 ? 15 - sunbtt_plus_batch_logn : 0;
      int nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      int log_nof_subntts_chunks = ntt.ntt_sub_logn.h1_layers_sub_logn[1] - log_nof_h1_subntts_todo_in_parallel;
      int nof_subntts_chunks = 1 << log_nof_subntts_chunks;
      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel;
             h1_subntt_idx_in_chunck++) {
          ntt_task_cordinates.h1_subntt_idx =
            h1_subntts_chunck_idx * nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;
          ntt.h1_cpu_ntt(output, ntt_task_cordinates, ntt_tasks_manager);
        }
        ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 0);
      }

      ntt.refactor_and_reorder(output, twiddles);
      ntt_task_cordinates.h1_layer_idx = 1;
      sunbtt_plus_batch_logn = ntt.ntt_sub_logn.h1_layers_sub_logn[1] + int(log2(config.batch_size));
      log_nof_h1_subntts_todo_in_parallel = sunbtt_plus_batch_logn < 15 ? 15 - sunbtt_plus_batch_logn : 0;
      nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      log_nof_subntts_chunks = ntt.ntt_sub_logn.h1_layers_sub_logn[0] - log_nof_h1_subntts_todo_in_parallel;
      nof_subntts_chunks = 1 << log_nof_subntts_chunks;
      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel;
             h1_subntt_idx_in_chunck++) {
          ntt_task_cordinates.h1_subntt_idx =
            h1_subntts_chunck_idx * nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;
          ntt.h1_cpu_ntt(output, ntt_task_cordinates, ntt_tasks_manager);
        }
        ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 1);
      }
      ntt_task_cordinates.h1_subntt_idx = 0; // reset so that reorder_output will calculate the correct memory index
      if (config.columns_batch) {
        ntt.reorder_output(output, ntt_task_cordinates, true);
      } else {
        for (int b = 0; b < config.batch_size; b++) {
          ntt.reorder_output(output + b * size, ntt_task_cordinates, true);
        }
      }
    } else {
      ntt.h1_cpu_ntt(output, ntt_task_cordinates, ntt_tasks_manager);
      ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 0);
    }

    delete tasks_manager;

    if (direction == NTTDir::kInverse) { // TODO SHANIE - do that in parallel
      S inv_size = S::inv_log_size(logn);
      for (uint64_t i = 0; i < total_memory_size; ++i) {
        output[i] = output[i] * inv_size;
      }
      if (config.coset_gen != S::one()) { ntt.coset_mul(output, twiddles, coset_stride, arbitrary_coset); }
    }

    if (config.ordering == Ordering::kNR || config.ordering == Ordering::kRR) {
      ntt_task_cordinates = {0, 0, 0, 0, 0};
      ntt.reorder_by_bit_reverse(
        ntt_task_cordinates, output,
        true); // TODO - check if access the fixed indexes instead of reordering may be more efficient?
    }

    return eIcicleError::SUCCESS;
  }

  // template <typename S = scalar_t, typename E = scalar_t>
  // eIcicleError cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir direction, const NTTConfig<S>&
  // config, E* output)
  // {
  //   // count only how much time it takes for the multiplications
  //   const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
  //   for (int layer = 0; layer < 3; layer++) {
  //     for (int subntt_idx = 0; subntt_idx < 1024; subntt_idx++) {
  //       for (int cooley_tukey_step = 0; cooley_tukey_step < 5; cooley_tukey_step++) {
  //         for (int i = 0; i < 16; i++) {
  //           E v = input[subntt_idx + i] * twiddles[i];
  //           E u = input[subntt_idx + i + 16];
  //           output[subntt_idx*1024 + i] = u + v;
  //           output[subntt_idx*1024 + i + 16] = u - v;
  //         }
  //       }
  //     }
  //   }
  //   return eIcicleError::SUCCESS;
  // }

} // namespace ntt_cpu
