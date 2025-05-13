#ifndef ICICLE_BACKEND_CPU_UTIL_H
#define ICICLE_BACKEND_CPU_UTIL_H

#pragma once

#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "tasks_manager.h"
#include <cstdint>
#include <sys/types.h>
#include <vector>
#include "icicle/utils/log.h"


#include "taskflow/taskflow.hpp"
#include "icicle/program/program.h"
#include "cpu_program_executor.h"


#define CONFIG_NOF_THREADS_KEY  "n_threads"
using namespace field_config;
namespace icicle {

// extract the number of threads to run from config
inline int get_nof_workers(const VecOpsConfig& config)
{
  if (config.ext && config.ext->has(CONFIG_NOF_THREADS_KEY)) { return config.ext->get<int>(CONFIG_NOF_THREADS_KEY); }

  const int hw_threads = std::thread::hardware_concurrency();
  // Note: no need to account for the main thread in vec-ops since it's doing little work
  return std::max(1, hw_threads);
}

} // namespace icicle

#endif // ICICLE_BACKEND_CPU_UTIL_H
