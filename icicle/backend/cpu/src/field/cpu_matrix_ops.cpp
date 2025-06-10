#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "tasks_manager.h"
#include <cstdint>
#include <sys/types.h>
#include <vector>

#include "taskflow/taskflow.hpp"
#include "icicle/program/program.h"
#include "cpu_program_executor.h"
#include "util.h"

using namespace field_config;
using namespace icicle;

/*********************************** MATRIX MULTIPLICATION ***********************************/

template <typename T>
static eIcicleError cpu_matrix_mult(
  const Device& device,
  const T* mat_a,
  uint32_t nof_rows_a,
  uint32_t nof_cols_a,
  const T* mat_b,
  uint32_t nof_rows_b,
  uint32_t nof_cols_b,
  const VecOpsConfig& config,
  T* mat_out)
{
  
  // Check for null pointers
  if (mat_a == nullptr || mat_b == nullptr || mat_out == nullptr) {
    return eIcicleError::INVALID_ARGUMENT;
  }

  // Check for zero dimensions
  if (nof_rows_a == 0 || nof_cols_a == 0 || nof_rows_b == 0 || nof_cols_b == 0) {
    return eIcicleError::INVALID_ARGUMENT;
  }

  // Check if inner dimensions match for matrix multiplication
  if (nof_cols_a != nof_rows_b) {
    return eIcicleError::INVALID_ARGUMENT;
  }
  
  const uint64_t stride = config.columns_batch ? config.batch_size : 1;
  const uint64_t total_elements_one_mat = static_cast<uint64_t>(nof_rows_a) * nof_cols_b;
  
  
  // Naive algorithm using taskflow


  // Divide the problem among workers
  const int nof_workers = get_nof_workers(config);
  const uint32_t rows_per_task = std::max(1U, (nof_rows_a + nof_workers - 1) / nof_workers); // ceil division
  
  tf::Taskflow taskflow; // Accumulate tasks
  tf::Executor executor; // execute all tasks accumulated on multiple threads
  
  // For each batch
  for (uint32_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++) {
    // Calculate pointers for current batch
    const T* curr_mat_a = config.columns_batch ? 
                          mat_a + batch_idx : 
                          mat_a + batch_idx * nof_rows_a * nof_cols_a;
    const T* curr_mat_b = config.columns_batch ? 
                          mat_b + batch_idx : 
                          mat_b + batch_idx * nof_rows_b * nof_cols_b;
    T* curr_mat_out = config.columns_batch ? 
                      mat_out + batch_idx : 
                      mat_out + batch_idx * total_elements_one_mat;
    
    // Divide rows among tasks
    for (uint32_t row_start = 0; row_start < nof_rows_a; row_start += rows_per_task) {
      uint32_t row_end = std::min(row_start + rows_per_task, nof_rows_a);
      
      taskflow.emplace([=]() {
        // Compute a block of the output matrix
        for (uint32_t i = row_start; i < row_end; i++) {
          for (uint32_t j = 0; j < nof_cols_b; j++) {
            // Initialize result element to zero
            T sum = T::zero();
            
            // Compute dot product of row i from A and column j from B
            for (uint32_t k = 0; k < nof_cols_a; k++) {
              uint64_t a_idx = config.columns_batch ? 
                              (i * nof_cols_a + k) * stride : 
                              i * nof_cols_a + k;
              uint64_t b_idx = config.columns_batch ? 
                              (k * nof_cols_b + j) * stride : 
                              k * nof_cols_b + j;
              
              sum = sum + curr_mat_a[a_idx] * curr_mat_b[b_idx];
            }
            
            // Store result
            uint64_t out_idx = config.columns_batch ? 
                              (i * nof_cols_b + j) * stride : 
                              i * nof_cols_b + j;
            curr_mat_out[out_idx] = sum;
          }
        }
      });
    }
  }
  
  // Run all tasks and wait for completion
  executor.run(taskflow).wait();
  taskflow.clear();
  return eIcicleError::SUCCESS;
}

REGISTER_MATRIX_MULT_BACKEND("CPU", cpu_matrix_mult<scalar_t>);
#ifdef RING
REGISTER_POLY_RING_MATRIX_MULT_BACKEND("CPU", cpu_matrix_mult<PolyRing>);

#endif
