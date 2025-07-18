
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

using namespace field_config;
using namespace icicle;

/* Enumeration for the selected operation to execute.
 * The worker task is templated by this enum and based on that the functionality is selected. */
enum VecOperation {
  VECTOR_ADD,
  VECTOR_SUB,
  VECTOR_MUL,
  VECTOR_DIV,
  VECTOR_INV,
  CONVERT_TO_MONTGOMERY,
  CONVERT_FROM_MONTGOMERY,
  VECTOR_SUM,
  VECTOR_PRODUCT,
  SCALAR_ADD_VEC,
  SCALAR_SUB_VEC,
  SCALAR_MUL_VEC,
  BIT_REVERSE,
  SLICE,

  NOF_VECTOR_OPERATIONS
};

/**
 * @class VectorOpTask
 * @brief Contains all the functionality that a single worker can execute for any vector operation.
 *
 * The enum VecOperation defines which functionality to execute.
 * Based on the enum value, the functionality is selected and the worker execute that function for every task that
 * dispatched by the manager.
 */
template <typename T, typename U>
class VectorOpTask : public TaskBase
{
public:
  // Constructor
  VectorOpTask() : TaskBase() {}

  // Set the operands to execute a task of 2 operands and 1 output and dispatch the task
  void send_2ops_task(
    VecOperation operation,
    const uint32_t nof_operations,
    const T* op_a,
    const U* op_b,
    const uint32_t stride,
    T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_op_b = op_b;
    m_stride = stride;
    m_output = output;
    dispatch();
  }

  // Set the operands to execute a task of 1 operand and 1 output and dispatch the task
  void send_1op_task(VecOperation operation, const uint32_t nof_operations, const T* op_a, T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_output = output;
    dispatch();
  }
  // Set the operands to execute a task of 1 operand and dispatch the task
  void
  send_intermidiate_res_task(VecOperation operation, const uint64_t stop_index, const T* op_a, const uint64_t stride)
  {
    m_operation = operation;
    m_stop_index = stop_index;
    m_op_a = op_a;
    m_stride = stride;
    dispatch();
  }

  // Set the operands for bit_reverse operation and dispatch the task
  void send_bit_reverse_task(
    VecOperation operation,
    uint32_t bit_size,
    uint64_t start_index,
    const uint32_t nof_operations,
    const T* op_a,
    const uint64_t stride,
    T* output)
  {
    m_operation = operation;
    m_bit_size = bit_size;
    m_start_index = start_index;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_stride = stride;
    m_output = output;
    dispatch();
  }

  // Set the operands for slice operation and dispatch the task
  void send_slice_task(
    VecOperation operation,
    uint64_t stride,
    uint64_t stride_out,
    const uint32_t nof_operations,
    const T* op_a,
    T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_output = output;
    m_stride = stride;
    m_stride_out = stride_out;
    dispatch();
  }

  // Execute the selected function based on m_operation
  virtual void execute() { (this->*functionPtrs[static_cast<size_t>(m_operation)])(); }

private:
  // Single worker functionality to execute vector add (+)
  void vector_add()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i] + m_op_b[i];
    }
  }

  // Single worker functionality to execute vector add (+)
  void vector_sub()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i] - m_op_b[i];
    }
  }
  // Single worker functionality to execute vector mul (*)
  void vector_mul()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i] * m_op_b[i];
    }
  }
  // Single worker functionality to execute vector div (/)
  void vector_div()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i] * m_op_b[i].inverse();
    }
  }
  // Single worker functionality to execute vector inv (^-1)
  void vector_inv()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i].inverse();
    }
  }
  // Single worker functionality to execute conversion from barret to montgomery
  void convert_to_montgomery()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i].to_montgomery();
    }
  }
  // Single worker functionality to execute conversion from montgomery to barret
  void convert_from_montgomery()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i].from_montgomery();
    }
  }
  // Single worker functionality to execute sum(vector)
  void vector_sum()
  {
    m_intermidiate_res = T::zero();
    for (uint64_t i = 0; i < (m_stop_index * m_stride); i = i + m_stride) {
      m_intermidiate_res = m_intermidiate_res + m_op_a[i];
    }
  }
  // Single worker functionality to execute product(vector)
  void vector_product()
  {
    m_intermidiate_res = T::one();
    for (uint64_t i = 0; i < (m_stop_index * m_stride); i = i + m_stride) {
      m_intermidiate_res = m_intermidiate_res * m_op_a[i];
    }
  }
  // Single worker functionality to execute scalar + vector
  void scalar_add_vec()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[m_stride * i] = *m_op_a + m_op_b[m_stride * i];
    }
  }
  // Single worker functionality to execute scalar - vector
  void scalar_sub_vec()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[m_stride * i] = *m_op_a - m_op_b[m_stride * i];
    }
  }
  // Single worker functionality to execute scalar * vector
  void scalar_mul_vec()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[m_stride * i] = *m_op_a * m_op_b[m_stride * i];
    }
  }
  // Single worker functionality to execute bit reverse reorder
  void bit_reverse()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      uint64_t idx = m_start_index + i;     // original index
      uint64_t rev_idx = m_start_index + i; // reverse index
      // Bit reverse the iundex for 64 bits
      rev_idx = ((rev_idx >> 1) & 0x5555555555555555) | ((rev_idx & 0x5555555555555555) << 1); // bit rev single bits
      rev_idx = ((rev_idx >> 2) & 0x3333333333333333) | ((rev_idx & 0x3333333333333333) << 2); // bit rev 2 bits chunk
      rev_idx = ((rev_idx >> 4) & 0x0F0F0F0F0F0F0F0F) | ((rev_idx & 0x0F0F0F0F0F0F0F0F) << 4); // bit rev 4 bits chunk
      rev_idx = ((rev_idx >> 8) & 0x00FF00FF00FF00FF) | ((rev_idx & 0x00FF00FF00FF00FF) << 8); // bit rev 8 bits chunk
      rev_idx =
        ((rev_idx >> 16) & 0x0000FFFF0000FFFF) | ((rev_idx & 0x0000FFFF0000FFFF) << 16); // bit rev 16 bits chunk
      rev_idx = (rev_idx >> 32) | (rev_idx << 32);                                       // bit rev 32 bits chunk
      rev_idx = rev_idx >> (64 - m_bit_size);                                            // Align rev_idx to the LSB

      if (m_output == m_op_a) { // inplace calculation
        if (rev_idx < idx) {    // only on of the threads need to work
          std::swap(m_output[m_stride * idx], m_output[m_stride * rev_idx]);
        }
      } else {                                                 // out of place calculation
        m_output[m_stride * idx] = m_op_a[m_stride * rev_idx]; // set index value
      }
    }
  }

  // Single worker functionality to execute slice
  void slice()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i * m_stride_out] = m_op_a[i * m_stride];
    }
  }

  // An array of available function pointers arranged according to the VecOperation enum
  using FunctionPtr = void (VectorOpTask::*)();
  static constexpr std::array<FunctionPtr, static_cast<int>(NOF_VECTOR_OPERATIONS)> functionPtrs = {
    &VectorOpTask::vector_add,              // VECTOR_ADD,
    &VectorOpTask::vector_sub,              // VECTOR_SUB,
    &VectorOpTask::vector_mul,              // VECTOR_MUL,
    &VectorOpTask::vector_div,              // VECTOR_DIV,
    &VectorOpTask::vector_inv,              // VECTOR_INV,
    &VectorOpTask::convert_to_montgomery,   // CONVERT_TO_MONTGOMERY,
    &VectorOpTask::convert_from_montgomery, // CONVERT_FROM_MONTGOMERY,
    &VectorOpTask::vector_sum,              // VECTOR_SUM
    &VectorOpTask::vector_product,          // VECTOR_PRODUCT
    &VectorOpTask::scalar_add_vec,          // SCALAR_ADD_VEC,
    &VectorOpTask::scalar_sub_vec,          // SCALAR_SUB_VEC,
    &VectorOpTask::scalar_mul_vec,          // SCALAR_MUL_VEC,
    &VectorOpTask::bit_reverse,             // BIT_REVERSE
    &VectorOpTask::slice,                   // SLICE

  };

  VecOperation m_operation;  // the operation to execute
  uint32_t m_nof_operations; // number of operations to execute for this task
  const T* m_op_a;           // pointer to operand A. Operand A is a vector, or matrix in case of replace_elements
  const U* m_op_b;           // pointer to operand B. Operand B is a vector or scalar
  uint64_t m_start_index;    // index used in bitreverse operation and out of place matrix transpose
  uint64_t m_stop_index;     // index used in reduce operations and out of place matrix transpose
  uint32_t m_bit_size;       // use in bitrev operation
  uint64_t m_stride;         // used to support column batch operations
  uint64_t m_stride_out;     // used in slice operation
  T*
    m_output; // pointer to the output. Can be a vector, scalar pointer, or a matrix pointer in case of replace_elements

public:
  T m_intermidiate_res;    // pointer to the output. Can be a vector or scalar pointer
  uint64_t m_idx_in_batch; // index in the batch. Used in intermediate res tasks
};

#define NOF_OPERATIONS_PER_TASK 512
#define CONFIG_NOF_THREADS_KEY  "n_threads"

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config)
{
  if (config.ext && config.ext->has(CONFIG_NOF_THREADS_KEY)) { return config.ext->get<int>(CONFIG_NOF_THREADS_KEY); }

  const int hw_threads = std::thread::hardware_concurrency();
  // Note: no need to account for the main thread in vec-ops since it's doing little work
  return std::max(1, hw_threads);
}

// Execute a full task from the type vector = vector (op) vector
template <typename T, typename U>
eIcicleError
cpu_2vectors_op(VecOperation op, const T* vec_a, const U* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T, U>> task_manager(get_nof_workers(config));
  const uint64_t total_nof_operations = size * config.batch_size;
  for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T, U>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_2ops_task(
      op, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i), vec_a + i, vec_b + i, 1, output + i);
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

// Execute a full task from the type vector = scalar (op) vector
template <typename T>
eIcicleError cpu_scalar_vector_op(
  VecOperation op, const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T, T>> task_manager(get_nof_workers(config));
  const uint64_t total_nof_operations = size;
  const uint32_t stride = config.columns_batch ? config.batch_size : 1;
  for (uint32_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
      VectorOpTask<T, T>* task_p = task_manager.get_idle_or_completed_task();
      task_p->send_2ops_task(
        op, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i), scalar_a + idx_in_batch,
        config.columns_batch ? vec_b + idx_in_batch + i * config.batch_size : vec_b + idx_in_batch * size + i, stride,
        config.columns_batch ? output + idx_in_batch + i * config.batch_size : output + idx_in_batch * size + i);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

///////////////////////////////////////////////////////
// Functions to register at the CPU backend
/*********************************** ADD ***********************************/
template <typename T>
eIcicleError cpu_vector_add(
  const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_ADD, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_ADD_BACKEND("CPU", cpu_vector_add<scalar_t>);

/*********************************** ACCUMULATE ***********************************/
template <typename T>
eIcicleError
cpu_vector_accumulate(const Device& device, T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config)
{
  return cpu_2vectors_op(VecOperation::VECTOR_ADD, vec_a, vec_b, size, config, vec_a);
}

REGISTER_VECTOR_ACCUMULATE_BACKEND("CPU", cpu_vector_accumulate<scalar_t>);

/*********************************** SUB ***********************************/
template <typename T>
eIcicleError cpu_vector_sub(
  const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_SUB, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_SUB_BACKEND("CPU", cpu_vector_sub<scalar_t>);

/*********************************** MUL ***********************************/
template <typename T, typename U>
eIcicleError cpu_vector_mul(
  const Device& device, const T* vec_a, const U* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_MUL, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_MUL_BACKEND("CPU", (cpu_vector_mul<scalar_t, scalar_t>));

/*********************************** DIV ***********************************/
template <typename T>
eIcicleError cpu_vector_div(
  const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_DIV, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_DIV_BACKEND("CPU", cpu_vector_div<scalar_t>);

/*********************************** INV ***********************************/
template <typename T>
eIcicleError cpu_vector_inv(const Device& device, const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_INV, vec_a, vec_a, size, config, output);
}

REGISTER_VECTOR_INV_BACKEND("CPU", cpu_vector_inv<scalar_t>);

/*********************************** CONVERT MONTGOMERY ***********************************/
template <typename T>
eIcicleError cpu_convert_montgomery(
  const Device& device, const T* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T, T>> task_manager(get_nof_workers(config));
  const uint64_t total_nof_operations = size * config.batch_size;
  for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T, T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_1op_task(
      (is_to_montgomery ? CONVERT_TO_MONTGOMERY : CONVERT_FROM_MONTGOMERY),
      std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i), input + i, output + i);
  }
  task_manager.wait_done();
  for (uint64_t i = 0; i < size * config.batch_size; i++) {}
  return eIcicleError::SUCCESS;
}

REGISTER_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_montgomery<scalar_t>);

/*********************************** SUM ***********************************/

template <typename T>
eIcicleError cpu_vector_sum(const Device& device, const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T, T>> task_manager(get_nof_workers(config));
  std::vector<bool> output_initialized = std::vector<bool>(config.batch_size, false);
  uint64_t vec_a_offset = 0;
  uint64_t idx_in_batch = 0;
  // run until all vector deployed and all tasks completed
  while (true) {
    VectorOpTask<T, T>* task_p =
      vec_a_offset < size ? task_manager.get_idle_or_completed_task() : task_manager.get_completed_task();
    if (task_p == nullptr) { return eIcicleError::SUCCESS; }
    if (task_p->is_completed()) {
      output[task_p->m_idx_in_batch] = output_initialized[task_p->m_idx_in_batch]
                                         ? output[task_p->m_idx_in_batch] + task_p->m_intermidiate_res
                                         : task_p->m_intermidiate_res;
      output_initialized[task_p->m_idx_in_batch] = true;
    }
    if (vec_a_offset < size) {
      task_p->m_idx_in_batch = idx_in_batch;
      task_p->send_intermidiate_res_task(
        VecOperation::VECTOR_SUM, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size - vec_a_offset),
        config.columns_batch ? vec_a + idx_in_batch + vec_a_offset * config.batch_size
                             : vec_a + idx_in_batch * size + vec_a_offset,
        config.columns_batch ? config.batch_size : 1);
      idx_in_batch++;
      if (idx_in_batch == config.batch_size) {
        vec_a_offset += NOF_OPERATIONS_PER_TASK;
        idx_in_batch = 0;
      }
    } else {
      task_p->set_idle();
    }
  }
}

REGISTER_VECTOR_SUM_BACKEND("CPU", cpu_vector_sum<scalar_t>);

/*********************************** PRODUCT ***********************************/
template <typename T>
eIcicleError
cpu_vector_product(const Device& device, const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T, T>> task_manager(get_nof_workers(config));
  std::vector<bool> output_initialized = std::vector<bool>(config.batch_size, false);
  uint64_t vec_a_offset = 0;
  uint64_t idx_in_batch = 0;
  // run until all vector deployed and all tasks completed
  while (true) {
    VectorOpTask<T, T>* task_p =
      vec_a_offset < size ? task_manager.get_idle_or_completed_task() : task_manager.get_completed_task();
    if (task_p == nullptr) { return eIcicleError::SUCCESS; }
    if (task_p->is_completed()) {
      output[task_p->m_idx_in_batch] = output_initialized[task_p->m_idx_in_batch]
                                         ? output[task_p->m_idx_in_batch] * task_p->m_intermidiate_res
                                         : task_p->m_intermidiate_res;
      output_initialized[task_p->m_idx_in_batch] = true;
    }
    if (vec_a_offset < size) {
      task_p->m_idx_in_batch = idx_in_batch;
      task_p->send_intermidiate_res_task(
        VecOperation::VECTOR_PRODUCT, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size - vec_a_offset),
        config.columns_batch ? vec_a + idx_in_batch + vec_a_offset * config.batch_size
                             : vec_a + idx_in_batch * size + vec_a_offset,
        config.columns_batch ? config.batch_size : 1);
      idx_in_batch++;
      if (idx_in_batch == config.batch_size) {
        vec_a_offset += NOF_OPERATIONS_PER_TASK;
        idx_in_batch = 0;
      }
    } else {
      task_p->set_idle();
    }
  }
}

REGISTER_VECTOR_PRODUCT_BACKEND("CPU", cpu_vector_product<scalar_t>);

/*********************************** Scalar + Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_add(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_ADD_VEC, scalar_a, vec_b, size, config, output);
}

REGISTER_SCALAR_ADD_VEC_BACKEND("CPU", cpu_scalar_add<scalar_t>);

/*********************************** Scalar - Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_sub(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_SUB_VEC, scalar_a, vec_b, size, config, output);
}

REGISTER_SCALAR_SUB_VEC_BACKEND("CPU", cpu_scalar_sub<scalar_t>);

/*********************************** MUL BY SCALAR***********************************/
template <typename T>
eIcicleError cpu_scalar_mul(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_MUL_VEC, scalar_a, vec_b, size, config, output);
}

REGISTER_SCALAR_MUL_VEC_BACKEND("CPU", cpu_scalar_mul<scalar_t>);

/*********************************** BIT REVERSE ***********************************/
template <typename T>
eIcicleError
cpu_bit_reverse(const Device& device, const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out)
{
  ICICLE_ASSERT(vec_in && vec_out && size != 0) << "Invalid argument";

  uint32_t logn = static_cast<uint32_t>(std::floor(std::log2(size)));
  ICICLE_ASSERT((1ULL << logn) == size) << "Invalid argument - size is not a power of 2";

  // Perform the bit reverse
  TasksManager<VectorOpTask<T, T>> task_manager(get_nof_workers(config));
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    for (uint64_t i = 0; i < size; i += NOF_OPERATIONS_PER_TASK) {
      VectorOpTask<T, T>* task_p = task_manager.get_idle_or_completed_task();

      task_p->send_bit_reverse_task(
        BIT_REVERSE, logn, i, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size - i),
        config.columns_batch ? vec_in + idx_in_batch : vec_in + idx_in_batch * size,
        config.columns_batch ? config.batch_size : 1,
        config.columns_batch ? vec_out + idx_in_batch : vec_out + idx_in_batch * size);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

REGISTER_BIT_REVERSE_BACKEND("CPU", cpu_bit_reverse<scalar_t>);

/*********************************** SLICE ***********************************/

template <typename T>
eIcicleError cpu_slice(
  const Device& device,
  const T* vec_in,
  uint64_t offset,
  uint64_t stride,
  uint64_t size_in,
  uint64_t size_out,
  const VecOpsConfig& config,
  T* vec_out)
{
  ICICLE_ASSERT(vec_in != nullptr && vec_out != nullptr) << "Error: Invalid argument - input or output vector is null";
  ICICLE_ASSERT(offset + (size_out - 1) * stride < size_in) << "Error: Invalid argument - slice out of bound";

  TasksManager<VectorOpTask<T, T>> task_manager(get_nof_workers(config));
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    for (uint64_t i = 0; i < size_out; i += NOF_OPERATIONS_PER_TASK) {
      VectorOpTask<T, T>* task_p = task_manager.get_idle_or_completed_task();
      task_p->send_slice_task(
        SLICE, config.columns_batch ? stride * config.batch_size : stride, config.columns_batch ? config.batch_size : 1,
        std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size_out - i),
        config.columns_batch ? vec_in + idx_in_batch + (offset + i * stride) * config.batch_size
                             : vec_in + idx_in_batch * size_in + offset + i * stride,
        config.columns_batch ? vec_out + idx_in_batch + i * config.batch_size : vec_out + idx_in_batch * size_out + i);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

REGISTER_SLICE_BACKEND("CPU", cpu_slice<scalar_t>);

/*********************************** Highest non-zero idx ***********************************/
template <typename T>
eIcicleError cpu_highest_non_zero_idx_internal(
  const Device& device,
  const T* input,
  uint64_t size,
  const VecOpsConfig& config,
  int64_t* out_idx /*OUT*/,
  int32_t idx_in_batch_to_calc)
{
  ICICLE_ASSERT(input && out_idx && size != 0) << "Error: Invalid argument";
  uint64_t stride = config.columns_batch ? config.batch_size : 1;
  uint32_t start_idx = (idx_in_batch_to_calc == -1) ? 0 : idx_in_batch_to_calc;
  uint32_t end_idx = (idx_in_batch_to_calc == -1) ? config.batch_size : idx_in_batch_to_calc + 1;
  for (uint64_t idx_in_batch = start_idx; idx_in_batch < end_idx; ++idx_in_batch) {
    out_idx[idx_in_batch] = -1; // zero vector is considered '-1' since 0 would be zero in vec[0]
    const T* curr_input =
      config.columns_batch ? input + idx_in_batch : input + idx_in_batch * size; // Pointer to the current vector
    for (int64_t i = size - 1; i >= 0; --i) {
      if (curr_input[i * stride] != T::zero()) {
        out_idx[idx_in_batch] = i;
        break;
      }
    }
  }
  return eIcicleError::SUCCESS;
}

template <typename T>
eIcicleError cpu_highest_non_zero_idx(
  const Device& device, const T* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx /*OUT*/)
{
  return cpu_highest_non_zero_idx_internal(device, input, size, config, out_idx, -1);
}

REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND("CPU", cpu_highest_non_zero_idx<scalar_t>);

/*********************************** Execute program ***********************************/
template <typename T>
eIcicleError cpu_execute_program(
  const Device& device, std::vector<T*>& data, const Program<T>& program, uint64_t size, const VecOpsConfig& config)
{
  if (data.size() != program.m_nof_parameters) {
    ICICLE_LOG_ERROR << "Program has " << program.m_nof_parameters << " while data has " << data.size()
                     << " parameters";
    return eIcicleError::INVALID_ARGUMENT;
  }
  tf::Taskflow taskflow; // Accumulate tasks
  tf::Executor executor; // execute all tasks accumulated on multiple threads
  const uint64_t total_nof_operations = size * config.batch_size;

  // Divide the problem to workers
  const int nof_workers = get_nof_workers(config);
  const uint64_t worker_task_size = (total_nof_operations + nof_workers - 1) / nof_workers; // round up

  for (uint64_t start_idx = 0; start_idx < total_nof_operations; start_idx += worker_task_size) {
    taskflow.emplace([=]() {
      CpuProgramExecutor prog_executor(program);
      // init prog_executor to point to data vectors
      for (int param_idx = 0; param_idx < program.m_nof_parameters; ++param_idx) {
        prog_executor.m_variable_ptrs[param_idx] = &(data[param_idx][start_idx]);
      }

      const uint64_t task_size = std::min(worker_task_size, total_nof_operations - start_idx);
      // run over all task elements in the arrays and execute the program
      for (uint64_t i = 0; i < task_size; i++) {
        prog_executor.execute();
        // update the program pointers
        for (int param_idx = 0; param_idx < program.m_nof_parameters; ++param_idx) {
          (prog_executor.m_variable_ptrs[param_idx])++;
        }
      }
    });
  }

  executor.run(taskflow).wait();
  taskflow.clear();
  return eIcicleError::SUCCESS;
}

REGISTER_EXECUTE_PROGRAM_BACKEND("CPU", cpu_execute_program<scalar_t>);

/*********************************** Polynomial evaluation ***********************************/

template <typename T>
eIcicleError cpu_poly_eval(
  const Device& device,
  const T* coeffs,
  uint64_t coeffs_size,
  const T* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  T* evals /*OUT*/)
{
  ICICLE_ASSERT(coeffs && domain && evals && coeffs_size != 0 && domain_size != 0) << "Error: Invalid argument";
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  uint64_t stride = config.columns_batch ? config.batch_size : 1;
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    const T* curr_coeffs = config.columns_batch ? coeffs + idx_in_batch : coeffs + idx_in_batch * coeffs_size;
    T* curr_evals = config.columns_batch ? evals + idx_in_batch : evals + idx_in_batch * domain_size;
    for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx) {
      curr_evals[eval_idx * stride] = curr_coeffs[(coeffs_size - 1) * stride];
      for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
        curr_evals[eval_idx * stride] =
          curr_evals[eval_idx * stride] * domain[eval_idx] + curr_coeffs[coeff_idx * stride];
      }
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_POLYNOMIAL_EVAL("CPU", cpu_poly_eval<scalar_t>);

/*============================== polynomial division ==============================*/
template <typename T>
void school_book_division_step_cpu(T* r, T* q, const T* b, int deg_r, int deg_b, const T& lc_b_inv, uint32_t stride = 1)
{
  int64_t monomial = deg_r - deg_b; // monomial=1 is 'x', monomial=2 is x^2 etc.

  T lc_r = r[deg_r * stride];         // leading coefficient of r
  T monomial_coeff = lc_r * lc_b_inv; // lc_r / lc_b

  // adding monomial s to q (q=q+s)
  q[monomial * stride] = monomial_coeff;

  for (int i = monomial; i <= deg_r; ++i) {
    T b_coeff = b[(i - monomial) * stride];
    r[i * stride] = r[i * stride] - monomial_coeff * b_coeff;
  }
}

template <typename T>
eIcicleError cpu_poly_divide(
  const Device& device,
  const T* numerator,
  uint64_t numerator_size,
  const T* denominator,
  uint64_t denominator_size,
  const VecOpsConfig& config,
  T* q_out /*OUT*/,
  uint64_t q_size,
  T* r_out /*OUT*/,
  uint64_t r_size)
{
  uint32_t stride = config.columns_batch ? config.batch_size : 1;
  auto numerator_deg = std::make_unique<int64_t[]>(config.batch_size);
  auto denominator_deg = std::make_unique<int64_t[]>(config.batch_size);
  auto deg_r = std::make_unique<int64_t[]>(config.batch_size);
  cpu_highest_non_zero_idx(device, numerator, numerator_size, config, numerator_deg.get());
  cpu_highest_non_zero_idx(device, denominator, denominator_size, config, denominator_deg.get());
  memset(r_out, 0, sizeof(T) * (r_size * config.batch_size));
  memcpy(r_out, numerator, sizeof(T) * (numerator_size * config.batch_size));

  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    ICICLE_ASSERT(r_size >= numerator_deg[idx_in_batch] + 1)
      << "polynomial division expects r(x) size to be similar to numerator size and higher than numerator "
         "degree(x).\nr_size = "
      << r_size << ", numerator_deg[" << idx_in_batch << "] = " << numerator_deg[idx_in_batch];
    ICICLE_ASSERT(q_size >= (numerator_deg[idx_in_batch] - denominator_deg[idx_in_batch] + 1))
      << "polynomial division expects q(x) size to be at least deg(numerator)-deg(denominator)+1.\nq_size = " << q_size
      << ", numerator_deg[" << idx_in_batch << "] = " << numerator_deg[idx_in_batch] << ", denominator_deg["
      << idx_in_batch << "] = " << denominator_deg[idx_in_batch];
    const T* curr_numerator =
      config.columns_batch ? numerator + idx_in_batch : numerator + idx_in_batch * numerator_size;
    const T* curr_denominator =
      config.columns_batch ? denominator + idx_in_batch : denominator + idx_in_batch * denominator_size;
    T* curr_q_out = config.columns_batch ? q_out + idx_in_batch : q_out + idx_in_batch * q_size;
    T* curr_r_out = config.columns_batch ? r_out + idx_in_batch : r_out + idx_in_batch * r_size;

    // invert largest coeff of b
    const T& lc_b_inv = curr_denominator[denominator_deg[idx_in_batch] * stride].inverse();
    deg_r[idx_in_batch] = numerator_deg[idx_in_batch];
    while (deg_r[idx_in_batch] >= denominator_deg[idx_in_batch]) {
      // each iteration is removing the largest monomial in r until deg(r)<deg(b)
      school_book_division_step_cpu(
        curr_r_out, curr_q_out, curr_denominator, deg_r[idx_in_batch], denominator_deg[idx_in_batch], lc_b_inv, stride);
      // compute degree of r
      cpu_highest_non_zero_idx_internal(device, r_out, deg_r[idx_in_batch], config, deg_r.get(), idx_in_batch);
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_POLYNOMIAL_DIVISION("CPU", cpu_poly_divide<scalar_t>);

#ifdef EXT_FIELD
REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND("CPU", cpu_bit_reverse<extension_t>);
REGISTER_SLICE_EXT_FIELD_BACKEND("CPU", cpu_slice<extension_t>);
REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND("CPU", cpu_vector_add<extension_t>);
REGISTER_VECTOR_ACCUMULATE_EXT_FIELD_BACKEND("CPU", cpu_vector_accumulate<extension_t>);
REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND("CPU", cpu_vector_sub<extension_t>);
REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND("CPU", (cpu_vector_mul<extension_t, extension_t>));
REGISTER_VECTOR_MIXED_MUL_BACKEND("CPU", (cpu_vector_mul<extension_t, scalar_t>));
REGISTER_VECTOR_DIV_EXT_FIELD_BACKEND("CPU", cpu_vector_div<extension_t>);
REGISTER_VECTOR_INV_EXT_FIELD_BACKEND("CPU", cpu_vector_inv<extension_t>);
REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND("CPU", cpu_convert_montgomery<extension_t>);
REGISTER_VECTOR_SUM_EXT_FIELD_BACKEND("CPU", cpu_vector_sum<extension_t>);
REGISTER_VECTOR_PRODUCT_EXT_FIELD_BACKEND("CPU", cpu_vector_product<extension_t>);
REGISTER_SCALAR_MUL_VEC_EXT_FIELD_BACKEND("CPU", cpu_scalar_mul<extension_t>);
REGISTER_SCALAR_ADD_VEC_EXT_FIELD_BACKEND("CPU", cpu_scalar_add<extension_t>);
REGISTER_SCALAR_SUB_VEC_EXT_FIELD_BACKEND("CPU", cpu_scalar_sub<extension_t>);
REGISTER_EXECUTE_PROGRAM_EXT_FIELD_BACKEND("CPU", cpu_execute_program<extension_t>);
#endif // EXT_FIELD

#ifdef RING
// Register APIs for rns type
REGISTER_BIT_REVERSE_RING_RNS_BACKEND("CPU", cpu_bit_reverse<scalar_rns_t>);
REGISTER_SLICE_RING_RNS_BACKEND("CPU", cpu_slice<scalar_rns_t>);
REGISTER_VECTOR_ADD_RING_RNS_BACKEND("CPU", cpu_vector_add<scalar_rns_t>);
REGISTER_VECTOR_ACCUMULATE_RING_RNS_BACKEND("CPU", cpu_vector_accumulate<scalar_rns_t>);
REGISTER_VECTOR_SUB_RING_RNS_BACKEND("CPU", cpu_vector_sub<scalar_rns_t>);
REGISTER_VECTOR_MUL_RING_RNS_BACKEND("CPU", (cpu_vector_mul<scalar_rns_t, scalar_rns_t>));
REGISTER_VECTOR_DIV_RING_RNS_BACKEND("CPU", cpu_vector_div<scalar_rns_t>);
REGISTER_VECTOR_INV_RING_RNS_BACKEND("CPU", cpu_vector_inv<scalar_rns_t>);
REGISTER_CONVERT_MONTGOMERY_RING_RNS_BACKEND("CPU", cpu_convert_montgomery<scalar_rns_t>);
REGISTER_VECTOR_SUM_RING_RNS_BACKEND("CPU", cpu_vector_sum<scalar_rns_t>);
REGISTER_VECTOR_PRODUCT_RING_RNS_BACKEND("CPU", cpu_vector_product<scalar_rns_t>);
REGISTER_SCALAR_MUL_VEC_RING_RNS_BACKEND("CPU", cpu_scalar_mul<scalar_rns_t>);
REGISTER_SCALAR_ADD_VEC_RING_RNS_BACKEND("CPU", cpu_scalar_add<scalar_rns_t>);
REGISTER_SCALAR_SUB_VEC_RING_RNS_BACKEND("CPU", cpu_scalar_sub<scalar_rns_t>);
REGISTER_EXECUTE_PROGRAM_RING_RNS_BACKEND("CPU", cpu_execute_program<scalar_rns_t>);

// RNS conversion
template <typename SrcType, typename DstType, bool into_rns>
eIcicleError
cpu_convert_rns(const Device& device, const SrcType* input, uint64_t size, const VecOpsConfig& config, DstType* output)
{
  tf::Taskflow taskflow;
  tf::Executor executor;
  const uint64_t total_nof_operations = size * config.batch_size;

  const int nof_workers = get_nof_workers(config);
  const uint64_t worker_task_size = (total_nof_operations + nof_workers - 1) / nof_workers; // round up

  for (uint64_t start_idx = 0; start_idx < total_nof_operations; start_idx += worker_task_size) {
    taskflow.emplace([=]() {
      const uint64_t end_idx = std::min(start_idx + worker_task_size, total_nof_operations);
      for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
        if constexpr (into_rns) {
          DstType::convert_direct_to_rns(&input[idx].limbs_storage, &output[idx].limbs_storage);
        } else {
          SrcType::convert_rns_to_direct(&input[idx].limbs_storage, &output[idx].limbs_storage);
        }
      }
    });
  }

  executor.run(taskflow).wait();
  taskflow.clear();
  return eIcicleError::SUCCESS;
}
REGISTER_CONVERT_TO_RNS_BACKEND("CPU", (cpu_convert_rns<scalar_t, scalar_rns_t, true /*into rns*/>));
REGISTER_CONVERT_FROM_RNS_BACKEND("CPU", (cpu_convert_rns<scalar_rns_t, scalar_t, false /*from rns*/>));
#endif // RING
