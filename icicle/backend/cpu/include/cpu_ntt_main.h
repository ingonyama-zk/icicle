#pragma once
#include "icicle/utils/log.h"
#include "ntt_cpu.h"
#include "ntt_cpu_non_parallel.h"
#include <iostream>

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
    ICICLE_ASSERT(!(size & (size - 1))) << "Size must be a power of 2. size = " << size;
    ICICLE_ASSERT(size <= CpuNttDomain<S>::s_ntt_domain.get_max_size())
      << "Size is too large for domain. size = " << size
      << ", domain_max_size = " << CpuNttDomain<S>::s_ntt_domain.get_max_size();
    uint32_t log_size = uint32_t(log2(size));
    uint32_t log_batch_size = uint32_t(log2(config.batch_size));
    uint32_t scalar_size = sizeof(S);

    bool parallel = is_parallel(log_size, log_batch_size, scalar_size);

    if (!parallel) {
      NttCpuNonParallel<S, E> ntt(log_size, direction, config, input, output);
      ntt.run();
    } else {
      NttCpu<S, E> ntt(log_size, direction, config, input, output);
      ntt.run();
    }
    return eIcicleError::SUCCESS;
  }
} // namespace ntt_cpu
