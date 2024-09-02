#pragma once
#include "ntt_tasks.h"

using namespace field_config;
using namespace icicle;

/**
 * @brief Performs the Number Theoretic Transform (NTT) on the input data.
 *
 * This function executes the NTT or inverse NTT on the given input data, managing
 * tasks and reordering elements as needed. It handles coset multiplications, task
 * hierarchy, and memory management for efficient computation.
 *
 * The NTT problem is given at a specific size and is divided into subproblems to enable
 * parallel solving of independent tasks, ensuring that the number of problems solved
 * simultaneously does not exceed cache size. The original problem is divided into hierarchies
 * of subproblems. Beyond a certain size, the problem is divided into two layers of sub-NTTs in
 * hierarchy 1. Within hierarchy 1, the problem is further divided into 1-3 layers of sub-NTTs
 * belonging to hierarchy 0. The division into hierarchies and the sizes of the sub-NTTs are
 * determined by the original problem size.
 *
 * The sub-NTTs within hierarchy 0 are the units of work that are assigned to individual threads.
 * The overall computation is executed in a multi-threaded fashion, with the degree of parallelism
 * determined by the number of available hardware cores.
 *
 * @param device The device on which the NTT is being performed.
 * @param input Pointer to the input data.
 * @param size The size of the input data, must be a power of 2.
 * @param direction The direction of the NTT (forward or inverse).
 * @param config Configuration settings for the NTT operation.
 * @param output Pointer to the output data.
 *
 * @return eIcicleError Status of the operation, indicating success or failure.
 */
namespace ntt_cpu {
  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError
  cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir direction, const NTTConfig<S>& config, E* output)
  {
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
    const uint64_t total_input_size = size * config.batch_size;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    NttCpu<S, E> ntt(logn, direction, config, domain_max_size, twiddles);
    NttTaskCordinates ntt_task_cordinates = {0, 0, 0, 0, 0};
    NttTasksManager<S, E> ntt_tasks_manager(logn);
    const int nof_threads = std::thread::hardware_concurrency();
    auto tasks_manager = new TasksManager<NttTask<S, E>>(nof_threads - 1);
    NttTask<S, E>* task_slot;
    std::unique_ptr<S[]> arbitrary_coset = nullptr;
    const int coset_stride = ntt.find_or_generate_coset(arbitrary_coset);

    ntt.copy_and_reorder_if_needed(input, output);
    if (config.coset_gen != S::one() && direction == NTTDir::kForward) {
      ntt.coset_mul(output, coset_stride, arbitrary_coset);
    }

    int sunbtt_plus_batch_logn;
    int log_nof_hierarchy_1_subntts_todo_in_parallel;
    int nof_hierarchy_1_subntts_todo_in_parallel;
    int log_nof_subntts_chunks;
    int nof_subntts_chunks;

    if (logn > HIERARCHY_1) {
      for (ntt_task_cordinates.hierarchy_1_layer_idx = 0; ntt_task_cordinates.hierarchy_1_layer_idx < 2;
           ntt_task_cordinates.hierarchy_1_layer_idx++) {
        sunbtt_plus_batch_logn =
          ntt.ntt_sub_logn.hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx] +
          int(log2(config.batch_size));
        log_nof_hierarchy_1_subntts_todo_in_parallel =
          sunbtt_plus_batch_logn < HIERARCHY_1 ? HIERARCHY_1 - sunbtt_plus_batch_logn : 0;
        nof_hierarchy_1_subntts_todo_in_parallel = 1 << log_nof_hierarchy_1_subntts_todo_in_parallel;
        log_nof_subntts_chunks =
          ntt.ntt_sub_logn.hierarchy_1_layers_sub_logn[1 - ntt_task_cordinates.hierarchy_1_layer_idx] -
          log_nof_hierarchy_1_subntts_todo_in_parallel;
        nof_subntts_chunks = 1 << log_nof_subntts_chunks;
        for (int hierarchy_1_subntts_chunck_idx = 0; hierarchy_1_subntts_chunck_idx < nof_subntts_chunks;
             hierarchy_1_subntts_chunck_idx++) {
          for (int hierarchy_1_subntt_idx_in_chunck = 0;
               hierarchy_1_subntt_idx_in_chunck < nof_hierarchy_1_subntts_todo_in_parallel;
               hierarchy_1_subntt_idx_in_chunck++) {
            ntt_task_cordinates.hierarchy_1_subntt_idx =
              hierarchy_1_subntts_chunck_idx * nof_hierarchy_1_subntts_todo_in_parallel +
              hierarchy_1_subntt_idx_in_chunck;
            ntt.hierarchy1_push_tasks(output, ntt_task_cordinates, ntt_tasks_manager);
          }
          ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, ntt_task_cordinates.hierarchy_1_layer_idx);
        }
        if (ntt_task_cordinates.hierarchy_1_layer_idx == 0) { ntt.hierarchy_1_reorder(output); }
      }
      // reset hierarchy_1_subntt_idx so that reorder_and_refactor_if_needed will calculate the correct memory index
      ntt_task_cordinates.hierarchy_1_subntt_idx = 0;
      if (config.columns_batch) {
        ntt.reorder_and_refactor_if_needed(output, ntt_task_cordinates, true);
      } else {
        for (int b = 0; b < config.batch_size; b++) {
          ntt.reorder_and_refactor_if_needed(output + b * size, ntt_task_cordinates, true);
        }
      }
    } else {
      ntt.hierarchy1_push_tasks(output, ntt_task_cordinates, ntt_tasks_manager);
      ntt.handle_pushed_tasks(tasks_manager, ntt_tasks_manager, 0);
    }

    if (direction == NTTDir::kInverse) {
      S inv_size = S::inv_log_size(logn);
      for (uint64_t i = 0; i < total_input_size; ++i) {
        output[i] = output[i] * inv_size;
      }
      if (config.coset_gen != S::one()) { ntt.coset_mul(output, coset_stride, arbitrary_coset); }
    }

    if (config.ordering == Ordering::kNR || config.ordering == Ordering::kRR) {
      ntt_task_cordinates = {0, 0, 0, 0, 0};
      ntt.reorder_by_bit_reverse(ntt_task_cordinates, output, true);
    }
    delete tasks_manager;
    return eIcicleError::SUCCESS;
  }
} // namespace ntt_cpu
