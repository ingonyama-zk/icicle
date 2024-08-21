#pragma once
// #include "icicle/backend/ntt_backend.h"
// #include "icicle/errors.h"
// #include "icicle/runtime.h"
// #include "icicle/utils/log.h"
// #include "icicle/fields/field_config.h"
// #include "icicle/vec_ops.h"
#include "ntt_tasks.h"

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

  //main
  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir direction, const NTTConfig<S>& config, E* output)
  {
    if (size & (size - 1)) {
      ICICLE_LOG_ERROR << "Size must be a power of 2. size = " << size;
      return eIcicleError::INVALID_ARGUMENT;
    }
    const int logn = int(log2(size));
    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    NttCpu<S, E> ntt(logn, direction, config, domain_max_size, twiddles);
    NttTaskCordinates ntt_task_cordinates = {0, 0, 0, 0, 0};
    NttTasksManager <S, E> ntt_tasks_manager(logn);
    auto tasks_manager = new TasksManager<NttTask<S, E>>(std::thread::hardware_concurrency()-1);
    // auto tasks_manager = new TasksManager<NttTask<S, E>>(1);


    int coset_stride = 0;
    std::unique_ptr<S[]> arbitrary_coset = nullptr;
    if (config.coset_gen != S::one()) { // TODO SHANIE - implement more efficient way to find coset_stride
      try {
        coset_stride = CpuNttDomain<S>::s_ntt_domain.coset_index.at(config.coset_gen); //Coset generator found in twiddles
      } catch (const std::out_of_range& oor) { //Coset generator not found in twiddles. Calculating arbitrary coset
        arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset[0] = S::one();
        S coset_gen = direction == NTTDir::kForward ? config.coset_gen : S::inverse(config.coset_gen); // inverse for INTT
        for (int i = 1; i <= domain_max_size; i++) {
          arbitrary_coset[i] = arbitrary_coset[i - 1] * coset_gen;
        }
      }
    }
    uint64_t total_memory_size = size * config.batch_size;
    std::copy(input, input + total_memory_size, output);
    if (config.ordering == Ordering::kRN || config.ordering == Ordering::kRR) {
      ntt.reorder_by_bit_reverse(ntt_task_cordinates, output, true); // TODO - check if access the fixed indexes instead of reordering may be more efficient?
    }

    if (config.coset_gen != S::one() && direction == NTTDir::kForward) {
      ntt.coset_mul(
        output, twiddles, coset_stride, arbitrary_coset);
    }
    NttTask<S, E>* task_slot;
    
    if (logn > 15) {
      // ICICLE_LOG_DEBUG << "START NTTs logn: " << logn;
      ntt.reorder_input(output);
      int sunbtt_plus_batch_logn = ntt.ntt_sub_logn.h1_layers_sub_logn[0] + int(log2(config.batch_size));
      int log_nof_h1_subntts_todo_in_parallel = sunbtt_plus_batch_logn < 15? 15 - sunbtt_plus_batch_logn : 0;
      int nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      int log_nof_subntts_chunks = ntt.ntt_sub_logn.h1_layers_sub_logn[1] - log_nof_h1_subntts_todo_in_parallel;
      int nof_subntts_chunks = 1 << log_nof_subntts_chunks;
      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel; h1_subntt_idx_in_chunck++) {
          ntt_task_cordinates.h1_subntt_idx =  h1_subntts_chunck_idx*nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;
          ntt.h1_cpu_ntt(output, ntt_task_cordinates, ntt_tasks_manager);
        }
        ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 0);
      }

      
      // ntt_tasks_manager.wait_for_all_tasks();
      ntt.refactor_and_reorder(output, twiddles);
      ntt_task_cordinates.h1_layer_idx = 1;
      sunbtt_plus_batch_logn = ntt.ntt_sub_logn.h1_layers_sub_logn[1] + int(log2(config.batch_size));
      log_nof_h1_subntts_todo_in_parallel = sunbtt_plus_batch_logn < 15? 15 - sunbtt_plus_batch_logn : 0;
      nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      log_nof_subntts_chunks = ntt.ntt_sub_logn.h1_layers_sub_logn[0] - log_nof_h1_subntts_todo_in_parallel;
      nof_subntts_chunks = 1 << log_nof_subntts_chunks;
      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel; h1_subntt_idx_in_chunck++) {
          ntt_task_cordinates.h1_subntt_idx =  h1_subntts_chunck_idx*nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;
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
      if (config.coset_gen != S::one()) {
        ntt.coset_mul(output, twiddles, coset_stride, arbitrary_coset);
      }
    }

    if (config.ordering == Ordering::kNR || config.ordering == Ordering::kRR) {
      ntt_task_cordinates = {0, 0, 0, 0, 0};
      ntt.reorder_by_bit_reverse(ntt_task_cordinates, output, true); // TODO - check if access the fixed indexes instead of reordering may be more efficient?
    }

    return eIcicleError::SUCCESS;
  }
} // namespace ntt_cpu















  // template <typename S = scalar_t, typename E = scalar_t>
  // eIcicleError cpu_ntt_parallel_try(
  //   uint64_t size,
  //   uint64_t original_size,
  //   NTTDir dir,
  //   const NTTConfig<S>& config,
  //   E* output,
  //   const S* twiddles,
  //   const int domain_max_size = 0)
  // {
  //   const int logn = int(log2(size));
  //   std::vector<int> layers_sntt_log_size(
  //     std::begin(layers_subntt_log_size[logn]), std::end(layers_subntt_log_size[logn]));

  //   unsigned int max_nof_parallel_threads = std::thread::hardware_concurrency();
  //   std::cout << "Number of concurrent threads supported: " << max_nof_parallel_threads << std::endl;

  //   std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> start_times(max_nof_parallel_threads*3);
  //   std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> finish_times(max_nof_parallel_threads*3);
  //   // Assuming that NTT fits in the cache, so we split the NTT to layers and calculate them one after the other.
  //   // Subntts inside the same laye are calculate in parallel.
  //   // Sorting is not needed, since the elements needed for each subntt are close to each other in memory.
  //   // Instead of sorting, we are using the function idx_in_mem to calculate the memory index of each element.
  //   for (int layer = 0; layer < layers_sntt_log_size.size(); layer++) {
  //     #if PARALLEL
  //     std::vector<std::thread> threads;
  //     #endif
  //     if (layer == 0) {
  //       int nof_subntts =  1 << layers_sntt_log_size[1];
  //       int nof_blocks = 1 << layers_sntt_log_size[2];
  //       int subntts_per_thread = (nof_subntts + max_nof_parallel_threads - 1) / max_nof_parallel_threads;
  //       int nof_threads = nof_subntts / subntts_per_thread;
  //       ICICLE_LOG_DEBUG << "layer: "<< layer <<", number of threads: " << nof_threads << ", subntts per thread: " << subntts_per_thread << ", nof_subntts: " << nof_subntts << ", nof_blocks: " << nof_blocks;
  //       for (int block_idx = 0; block_idx < nof_blocks; block_idx++) {
  //         for (int thread_idx = 0; thread_idx < nof_threads; thread_idx++) {
  //           #if PARALLEL
  //           threads.emplace_back([&, block_idx, thread_idx] {
  //           #endif
  //             for (int subntt_idx = thread_idx; subntt_idx < nof_subntts; subntt_idx += nof_threads) {
  //               int idx = block_idx * nof_subntts + subntt_idx;
  //               ICICLE_LOG_DEBUG << "layer: "<< layer <<", block_idx: " << block_idx << ", subntt_idx: " << subntt_idx << ", idx: " << idx;
  //               start_times[idx] = std::chrono::high_resolution_clock::now();
  //               cpu_ntt_basic(
  //                 output, original_size, dir, config, output, block_idx, subntt_idx, layers_sntt_log_size, layer);
  //               finish_times[idx] = std::chrono::high_resolution_clock::now();
  //             }
  //           #if PARALLEL
  //           });
  //           #endif
  //         }
  //       }
  //       #if PARALLEL
  //       for (auto& th : threads) {
  //         th.join();
  //       }
  //       #endif
  //     }
  //     if (layer == 1 && layers_sntt_log_size[1]) {
  //       int nof_subntts = 1 << layers_sntt_log_size[0];
  //       int nof_blocks = 1 << layers_sntt_log_size[2];
  //       int subntts_per_thread = (nof_subntts + max_nof_parallel_threads - 1) / max_nof_parallel_threads;
  //       int nof_threads = nof_subntts / subntts_per_thread;
  //       ICICLE_LOG_DEBUG << "layer: "<< layer <<", number of threads: " << nof_threads << ", subntts per thread: " << subntts_per_thread << ", nof_subntts: " << nof_subntts << ", nof_blocks: " << nof_blocks;
  //       for (int block_idx = 0; block_idx < nof_blocks; block_idx++) {
  //         for (int thread_idx = 0; thread_idx < nof_threads; thread_idx++) {
  //           #if PARALLEL
  //           threads.emplace_back([&, block_idx, thread_idx] {
  //           #endif
  //             for (int subntt_idx = thread_idx; subntt_idx < nof_subntts; subntt_idx += nof_threads) {
  //               int idx = (1 << layers_sntt_log_size[1]) + block_idx * nof_subntts + subntt_idx;
  //               ICICLE_LOG_DEBUG << "layer: "<< layer <<", block_idx: " << block_idx << ", subntt_idx: " << subntt_idx << ", idx: " << idx;
  //               start_times[idx] = std::chrono::high_resolution_clock::now();
  //               cpu_ntt_basic(
  //                 output, original_size, dir, config, output, block_idx, subntt_idx, layers_sntt_log_size, layer);
  //               finish_times[idx] = std::chrono::high_resolution_clock::now();
  //             }
  //           #if PARALLEL
  //           });
  //           #endif
  //         }
  //       }
  //       #if PARALLEL
  //       for (auto& th : threads) {
  //         th.join();
  //       }
  //       #endif
  //     }
  //     if (layer == 2 && layers_sntt_log_size[2]) {
  //       int nof_blocks = 1 << (layers_sntt_log_size[0] + layers_sntt_log_size[1]);
  //       int subntts_per_thread = (nof_blocks + max_nof_parallel_threads - 1) / max_nof_parallel_threads;
  //       int nof_threads = nof_blocks / subntts_per_thread;
  //       for (int thread_idx = 0; thread_idx < nof_threads; thread_idx++) {
  //         #if PARALLEL
  //         threads.emplace_back([&, thread_idx] {
  //         #endif
  //           for (int block_idx = thread_idx; block_idx < nof_blocks; block_idx += nof_threads) {
  //             int idx = (1 << layers_sntt_log_size[1]) + (1 << layers_sntt_log_size[0]) + block_idx;
  //             start_times[idx] = std::chrono::high_resolution_clock::now();
  //             cpu_ntt_basic(
  //               output, original_size, dir, config, output, block_idx, 0/*subntt_idx - not used*/, layers_sntt_log_size, layer);
  //             finish_times[idx] = std::chrono::high_resolution_clock::now();
  //           }
  //         #if PARALLEL
  //         });
  //         #endif
  //       }
  //       #if PARALLEL
  //       for (auto& th : threads) {
  //         th.join();
  //       }
  //       #endif
  //     }
  //     // if (layer != 2 && layers_sntt_log_size[layer + 1] != 0) {
  //     //   refactor_output<S, E>(
  //     //     output, original_size, config.batch_size, config.columns_batch, twiddles,
  //     //     domain_max_size, layers_sntt_log_size, layer, dir);
  //     // }
  //   }
  //   // Sort the output at the end so that elements will be in right order.
  //   // TODO SHANIE  - After implementing for different ordering, maybe this should be done in a different place
  //   //              - When implementing real parallelism, consider sorting in parallel and in-place
  //   if (layers_sntt_log_size[1]) { // at least 2 layers
  //     if (config.columns_batch) {
  //       reorder_output(output, size, layers_sntt_log_size, config.batch_size, config.columns_batch);
  //     } else {
  //       for (int b = 0; b < config.batch_size; b++) {
  //         reorder_output(
  //           output + b * original_size, size, layers_sntt_log_size, config.batch_size, config.columns_batch);
  //       }
  //     }
  //   }
  //   // #if PARALLEL
  //   // Print start and finish times
  //   for (size_t i = 0; i < start_times.size(); ++i) {
  //       auto start_ns = std::chrono::duration_cast<std::chrono::microseconds>(start_times[i].time_since_epoch()).count();
  //       auto finish_ns = std::chrono::duration_cast<std::chrono::microseconds>(finish_times[i].time_since_epoch()).count();
  //       // std::cout << "thread " << i << " started at " << start_ns
  //       //   << " μs and finished at " << finish_ns << " μs. Total: " << (finish_ns - start_ns) << " μs\n";
  //   }
  //   // #endif
  //   return eIcicleError::SUCCESS;
  // }
  
