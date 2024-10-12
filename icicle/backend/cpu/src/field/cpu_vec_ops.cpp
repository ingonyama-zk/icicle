
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "tasks_manager.h"
#include <cstdint>
#include <sys/types.h>
#include <vector>

using namespace field_config;
using namespace icicle;

/* Enumeration for the selected operation to execute.
 * The worker task is templated by this enum and based on that the functionality is selected. */
enum VecOperation {
  VECTOR_ADD,
  VECTOR_SUB,
  VECTOR_MUL,
  VECTOR_DIV,
  CONVERT_TO_MONTGOMERY,
  CONVERT_FROM_MONTGOMERY,
  VECTOR_SUM,
  VECTOR_PRODUCT,
  SCALAR_ADD_VEC,
  SCALAR_SUB_VEC,
  SCALAR_MUL_VEC,
  BIT_REVERSE,
  SLICE,
  REPLACE_ELEMENTS,
  OUT_OF_PLACE_MATRIX_TRANSPOSE,

  NOF_OPERATIONS
};

/**
 * @class VectorOpTask
 * @brief Contains all the functionality that a single worker can execute for any vector operation.
 *
 * The enum VecOperation defines which functionality to execute.
 * Based on the enum value, the functionality is selected and the worker execute that function for every task that
 * dispatched by the manager.
 */
template <typename T>
class VectorOpTask : public TaskBase
{
public:
  // Constructor
  VectorOpTask() : TaskBase() {}

  // Set the operands to execute a task of 2 operands and 1 output and dispatch the task
  void send_2ops_task(VecOperation operation, const uint32_t nof_operations, const T* op_a, const T* op_b, const uint32_t stride , T* output)
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
  void send_intermidiate_res_task(VecOperation operation, const uint64_t stop_index, const T* op_a, const uint64_t stride)
  {
    m_operation = operation;
    m_stop_index = stop_index;
    m_op_a = op_a;
    m_stride = stride;
    dispatch();
  }

  // Set the operands for bit_reverse operation and dispatch the task
  void send_bit_reverse_task(
    VecOperation operation, uint32_t bit_size, uint64_t start_index, const uint32_t nof_operations, const T* op_a, const uint64_t stride, T* output)
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
  void send_slice_task(VecOperation operation, uint64_t stride, uint64_t stride_out, const uint32_t nof_operations, const T* op_a, T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_output = output;
    m_stride = stride;
    m_stride_out = stride_out;
    dispatch();
  }

  // Set the operands for replace_elements operation and dispatch the task
  void send_replace_elements_task(VecOperation operation, const T* mat_in, const uint32_t nof_operations, std::vector<uint64_t>& start_indices_in_mat, uint64_t start_index, uint32_t log_nof_rows, uint32_t log_nof_cols, const uint32_t stride, T* mat_out)
  {
    m_operation = operation;
    m_op_a = mat_in;
    m_nof_operations = nof_operations;
    m_start_indices_in_mat = &start_indices_in_mat;
    m_start_index = start_index; //start index in start_indices vector
    m_log_nof_rows = log_nof_rows;
    m_log_nof_cols = log_nof_cols;
    m_stride = stride;
    m_output = mat_out;
    dispatch();
  }

  void send_out_of_place_matrix_transpose_task(VecOperation operation, const T* mat_in, const uint32_t nof_operations, const uint32_t nof_rows, const uint32_t nof_cols, const uint32_t stride, T* mat_out)
    {
      m_operation = operation;
      m_op_a = mat_in;
      m_nof_operations = nof_operations;
      m_nof_rows = nof_rows;
      m_nof_cols = nof_cols;
      m_stride = stride;
      m_output = mat_out;
      dispatch();
    }

  // Execute the selected function based on m_operation
  virtual void execute() {
    (this->*functionPtrs[static_cast<size_t>(m_operation)])(); 
  }

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
      m_output[i] = m_op_a[i] * T::inverse(m_op_b[i]);
    }
  }
  // Single worker functionality to execute conversion from barret to montgomery
  void convert_to_montgomery()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = T::to_montgomery(m_op_a[i]);
    }
  }
  // Single worker functionality to execute conversion from montgomery to barret
  void convert_from_montgomery()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = T::from_montgomery(m_op_a[i]);
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
          std::swap(m_output[m_stride*idx], m_output[m_stride*rev_idx]);
        }
      } else {                           // out of place calculation
        m_output[m_stride*idx] = m_op_a[m_stride*rev_idx]; // set index value
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

  // Function to perform modulus with Mersenne number
  uint64_t mersenne_mod(uint64_t shifted_idx, uint32_t total_bits) {
    uint64_t mod = (1ULL << total_bits) - 1;
    shifted_idx = (shifted_idx & mod) + (shifted_idx >> total_bits);
    while (shifted_idx >= mod) {
      shifted_idx = (shifted_idx & mod) + (shifted_idx >> total_bits);
    }
    return shifted_idx;
  }


  // Single worker functionality to execute replace elements
  void replace_elements()
  {
    const uint32_t total_bits = m_log_nof_rows + m_log_nof_cols;
    for (uint32_t i = 0; i < m_nof_operations; ++i) {
      uint64_t start_idx = (*m_start_indices_in_mat)[m_start_index + i];
      uint64_t idx = start_idx;
        T prev = m_op_a[m_stride * idx];
      do {
        uint64_t shifted_idx = idx << m_log_nof_rows;
        uint64_t new_idx = mersenne_mod(shifted_idx, total_bits);
        T next = m_op_a[m_stride * new_idx];
        m_output[m_stride * new_idx] = prev;
        prev = next;
        idx = new_idx;
      } while (idx != start_idx);
    }
  }

  // Single worker functionality for out of palce matrix transpose
  void out_of_place_transpose()
  {
    for (uint32_t k = 0; k < m_nof_operations; ++k) {
      for (uint32_t j = 0; j < m_nof_cols; ++j) {
        m_output[m_stride * (j * m_nof_rows + k)] = m_op_a[m_stride * (k * m_nof_cols + j)];
      }
    }
  }



  // An array of available function pointers arranged according to the VecOperation enum
  using FunctionPtr = void (VectorOpTask::*)();
  static constexpr std::array<FunctionPtr, static_cast<int>(NOF_OPERATIONS)> functionPtrs = {
    &VectorOpTask::vector_add,              // VECTOR_ADD,
    &VectorOpTask::vector_sub,              // VECTOR_SUB,
    &VectorOpTask::vector_mul,              // VECTOR_MUL,
    &VectorOpTask::vector_div,              // VECTOR_DIV,
    &VectorOpTask::convert_to_montgomery,   // CONVERT_TO_MONTGOMERY,
    &VectorOpTask::convert_from_montgomery, // CONVERT_FROM_MONTGOMERY,
    &VectorOpTask::vector_sum,              // VECTOR_SUM
    &VectorOpTask::vector_product,          // VECTOR_PRODUCT
    &VectorOpTask::scalar_add_vec,          // SCALAR_ADD_VEC,
    &VectorOpTask::scalar_sub_vec,          // SCALAR_SUB_VEC,
    &VectorOpTask::scalar_mul_vec,          // SCALAR_MUL_VEC,
    &VectorOpTask::bit_reverse,             // BIT_REVERSE
    &VectorOpTask::slice,                   // SLICE
    &VectorOpTask::replace_elements,        // REPLACE_ELEMENTS
    &VectorOpTask::out_of_place_transpose   // OUT_OF_PLACE_MATRIX_TRANSPOSE


  };

  VecOperation m_operation; // the operation to execute
  uint32_t m_nof_operations;     // number of operations to execute for this task
  const T* m_op_a;          // pointer to operand A. Operand A is a vector, or metrix in case of replace_elements
  const T* m_op_b;          // pointer to operand B. Operand B is a vector or scalar
  uint64_t m_start_index;   // index used in bitreverse operation and out of place matrix transpose
  uint64_t m_stop_index;    // index used in reduce operations and out of place matrix transpose
  uint32_t m_bit_size;      // use in bitrev operation
  uint64_t m_stride;        // used to support column batch operations
  uint64_t m_stride_out;    // used in slice operation
  T* m_output;              // pointer to the output. Can be a vector, scalar pointer, or a matrix pointer in case of replace_elements
  uint32_t m_log_nof_rows;  // log of the number of rows in the matrix, used in replace_elements
  uint32_t m_log_nof_cols;  // log of the number of columns in the matrix, used in replace_elements
  uint32_t m_nof_rows;      // the number of rows in the matrix, used in out of place matrix transpose
  uint32_t m_nof_cols;      // the number of columns in the matrix, used in out of place matrix transpose
  const std::vector<uint64_t>* m_start_indices_in_mat; // Indices used in replace_elements operations

public:  
  T m_intermidiate_res;     // pointer to the output. Can be a vector or scalar pointer
  uint64_t m_idx_in_batch;    // index in the batch. Used in intermidiate res tasks
}; // class VectorOpTask

#define NOF_OPERATIONS_PER_TASK 512
#define CONFIG_NOF_THREADS_KEY  "n_threads"

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config)
{
  if (config.ext && config.ext->has(CONFIG_NOF_THREADS_KEY)) { return config.ext->get<int>(CONFIG_NOF_THREADS_KEY); }

  int hw_threads = std::thread::hardware_concurrency();
  return ((hw_threads > 1) ? hw_threads - 1 : 1); // reduce 1 for the main
}

// Execute a full task from the type vector = vector (op) vector
template <typename T>
eIcicleError
cpu_2vectors_op(VecOperation op, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  const uint64_t total_nof_operations = size*config.batch_size;
  for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_2ops_task(op, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i), vec_a + i, vec_b + i, 1, output + i);
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

// Execute a full task from the type vector = scalar (op) vector
template <typename T>
eIcicleError cpu_scalar_vector_op(
  VecOperation op, const T* scalar_a, const T* vec_b, uint64_t size, bool use_single_scalar, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  const uint64_t total_nof_operations = use_single_scalar? size*config.batch_size : size;
  const uint32_t stride = (!use_single_scalar && config.columns_batch)? config.batch_size : 1;
  for (uint32_t idx_in_batch = 0; idx_in_batch < (use_single_scalar? 1 : config.batch_size); idx_in_batch++) {
    for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
      VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
      task_p->send_2ops_task(
      op,
      std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i),
      scalar_a + idx_in_batch,
      (!use_single_scalar && config.columns_batch)? vec_b + idx_in_batch + i*config.batch_size : vec_b + idx_in_batch*size + i,
      stride,
      (!use_single_scalar && config.columns_batch)? output + idx_in_batch + i*config.batch_size : output + idx_in_batch*size + i);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

///////////////////////////////////////////////////////
// Functions to register at the CPU backend
/*********************************** ADD ***********************************/
template <typename T>
eIcicleError
cpu_vector_add(const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
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
eIcicleError
cpu_vector_sub(const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_SUB, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_SUB_BACKEND("CPU", cpu_vector_sub<scalar_t>);

/*********************************** MUL ***********************************/
template <typename T>
eIcicleError
cpu_vector_mul(const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_MUL, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_MUL_BACKEND("CPU", cpu_vector_mul<scalar_t>);

/*********************************** DIV ***********************************/
template <typename T>
eIcicleError
cpu_vector_div(const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_DIV, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_DIV_BACKEND("CPU", cpu_vector_div<scalar_t>);

/*********************************** CONVERT MONTGOMERY ***********************************/
template <typename T>
eIcicleError cpu_convert_montgomery(
  const Device& device, const T* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  const uint64_t total_nof_operations = size*config.batch_size;
  for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_1op_task(
      (is_to_montgomery ? CONVERT_TO_MONTGOMERY : CONVERT_FROM_MONTGOMERY), std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i),
      input + i, output + i);
  }
  task_manager.wait_done();
  for (uint64_t i = 0; i < size*config.batch_size; i++) {
  }
  return eIcicleError::SUCCESS;
}

REGISTER_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_montgomery<scalar_t>);

#ifdef EXT_FIELD
REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND("CPU", cpu_vector_add<extension_t>);
REGISTER_VECTOR_ACCUMULATE_EXT_FIELD_BACKEND("CPU", cpu_vector_accumulate<extension_t>);
REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND("CPU", cpu_vector_sub<extension_t>);
REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND("CPU", cpu_vector_mul<extension_t>);
REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND("CPU", cpu_convert_montgomery<extension_t>);
#endif // EXT_FIELD

/*********************************** SUM ***********************************/

template <typename T>
eIcicleError cpu_vector_sum(const Device& device, const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  std::vector<bool> output_initialized = std::vector<bool>(config.batch_size, false);
  uint64_t vec_a_offset = 0;
  uint64_t idx_in_batch = 0;
  // run until all vector deployed and all tasks completed
  while (true) {
    VectorOpTask<T>* task_p  = vec_a_offset < size ? task_manager.get_idle_or_completed_task() : task_manager.get_completed_task();
    if (task_p == nullptr) {
      return eIcicleError::SUCCESS;
    }
    if (task_p->is_completed()) {
      output[task_p->m_idx_in_batch] = output_initialized[task_p->m_idx_in_batch] ? output[task_p->m_idx_in_batch] + task_p->m_intermidiate_res : task_p->m_intermidiate_res;
      output_initialized[task_p->m_idx_in_batch] = true;
    }
    if (vec_a_offset < size) {
      task_p->m_idx_in_batch = idx_in_batch;
      task_p->send_intermidiate_res_task(
        VecOperation::VECTOR_SUM,
        std::min((uint64_t)NOF_OPERATIONS_PER_TASK , size - vec_a_offset),
        config.columns_batch? vec_a + idx_in_batch + vec_a_offset*config.batch_size : vec_a + idx_in_batch*size + vec_a_offset,
        config.columns_batch? config.batch_size : 1);
      idx_in_batch++;
      if (idx_in_batch == config.batch_size) {
        vec_a_offset += NOF_OPERATIONS_PER_TASK;
        idx_in_batch = 0;
      }
    }
    else {
      task_p->set_idle();
    }
  }
}

REGISTER_VECTOR_SUM_BACKEND("CPU", cpu_vector_sum<scalar_t>);

/*********************************** PRODUCT ***********************************/
template <typename T>
eIcicleError cpu_vector_product(const Device& device, const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  std::vector<bool> output_initialized = std::vector<bool>(config.batch_size, false);
  uint64_t vec_a_offset = 0;
  uint64_t idx_in_batch = 0;
  // run until all vector deployed and all tasks completed
  while (true) {
    VectorOpTask<T>* task_p  = vec_a_offset < size ? task_manager.get_idle_or_completed_task() : task_manager.get_completed_task();
    if (task_p == nullptr) {
      return eIcicleError::SUCCESS;
    }
    if (task_p->is_completed()) {
      output[task_p->m_idx_in_batch] = output_initialized[task_p->m_idx_in_batch] ? output[task_p->m_idx_in_batch] * task_p->m_intermidiate_res : task_p->m_intermidiate_res;
      output_initialized[task_p->m_idx_in_batch] = true;
    }
    if (vec_a_offset < size) {
      task_p->m_idx_in_batch = idx_in_batch;
      task_p->send_intermidiate_res_task(
        VecOperation::VECTOR_PRODUCT,
        std::min((uint64_t)NOF_OPERATIONS_PER_TASK , size - vec_a_offset),
        config.columns_batch? vec_a + idx_in_batch + vec_a_offset*config.batch_size : vec_a + idx_in_batch*size + vec_a_offset,
        config.columns_batch? config.batch_size : 1);
      idx_in_batch++;
      if (idx_in_batch == config.batch_size) {
        vec_a_offset += NOF_OPERATIONS_PER_TASK;
        idx_in_batch = 0;
      }
    }
    else {
      task_p->set_idle();
    }
  }
}

REGISTER_VECTOR_PRODUCT_BACKEND("CPU", cpu_vector_product<scalar_t>);

/*********************************** Scalar + Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_add(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, bool use_single_scalar, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_ADD_VEC, scalar_a, vec_b, size, use_single_scalar, config, output);
}

REGISTER_SCALAR_ADD_VEC_BACKEND("CPU", cpu_scalar_add<scalar_t>);

/*********************************** Scalar - Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_sub(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, bool use_single_scalar, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_SUB_VEC, scalar_a, vec_b, size, use_single_scalar, config, output);
}

REGISTER_SCALAR_SUB_VEC_BACKEND("CPU", cpu_scalar_sub<scalar_t>);

/*********************************** MUL BY SCALAR***********************************/
template <typename T>
eIcicleError cpu_scalar_mul(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, bool use_single_scalar, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_MUL_VEC, scalar_a, vec_b, size, use_single_scalar, config, output);
}

REGISTER_SCALAR_MUL_VEC_BACKEND("CPU", cpu_scalar_mul<scalar_t>);

/*********************************** TRANSPOSE ***********************************/

template <typename T>
eIcicleError out_of_place_matrix_transpose(
  const Device& device, const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  uint32_t stride = config.columns_batch? config.batch_size : 1;
  const uint64_t total_elements_one_mat = static_cast<uint64_t>(nof_rows) * nof_cols;
  const uint32_t NOF_ROWS_PER_TASK = std::min((uint64_t)nof_rows, std::max((uint64_t)(NOF_OPERATIONS_PER_TASK / nof_cols) , (uint64_t)1));
  for (uint32_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    const T* cur_mat_in = config.columns_batch? mat_in + idx_in_batch : mat_in + idx_in_batch * total_elements_one_mat;
    T* cur_mat_out = config.columns_batch? mat_out + idx_in_batch : mat_out + idx_in_batch * total_elements_one_mat;
    // Perform the matrix transpose
    for (uint32_t i = 0; i < nof_rows; i += NOF_ROWS_PER_TASK) {
      VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
      task_p->send_out_of_place_matrix_transpose_task(
        OUT_OF_PLACE_MATRIX_TRANSPOSE,
        cur_mat_in + stride*i*nof_cols,
        std::min((uint64_t)NOF_ROWS_PER_TASK, (uint64_t)nof_rows - i),
        nof_rows,
        nof_cols,
        stride,
        cur_mat_out + (stride * i));
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

// Recursive function to generate all k-ary necklaces and to replace the elements withing the necklaces
template <typename T>
void gen_necklace(uint32_t t, uint32_t p, uint32_t k, uint32_t length, std::vector<uint32_t>& necklace, std::vector<uint64_t>& task_indices) {
  if (t > length) {
    if (length % p == 0 && !std::all_of(necklace.begin() + 1, necklace.begin() + length + 1,[first_element = necklace[1]](uint32_t x) { return x == first_element; })) {
      uint32_t start_idx = 0;
      uint64_t multiplier = 1;
      for (int i = length; i >= 1; --i) { // Compute start_idx as the decimal representation of the necklace
        start_idx += necklace[i] * multiplier;
        multiplier *= k;
      }
      task_indices.push_back(start_idx);
    }
    return;
  }

  necklace[t] = necklace[t - p];
  gen_necklace<T>(t + 1, p, k, length, necklace, task_indices);

  for (int i = necklace[t - p] + 1; i < k; ++i) {
    necklace[t] = i;
    gen_necklace<T>(t + 1, t, k, length, necklace, task_indices);
  }
}

template <typename T>
eIcicleError matrix_transpose_necklaces(const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out){
  uint32_t log_nof_rows = static_cast<uint32_t>(std::floor(std::log2(nof_rows)));
  uint32_t log_nof_cols = static_cast<uint32_t>(std::floor(std::log2(nof_cols)));
  uint32_t gcd_value = gcd(log_nof_rows, log_nof_cols);
  uint32_t k = 1 << gcd_value; // Base of necklaces
  uint32_t length = (log_nof_cols + log_nof_rows) / gcd_value; // length of necklaces. Since all are powers of 2, equvalent to (log_nof_cols + log_nof_rows) / gcd_value;
  const uint64_t max_nof_operations = NOF_OPERATIONS_PER_TASK / length;
  const uint64_t total_elements_one_mat = static_cast<uint64_t>(nof_rows) * nof_cols;

  std::vector<uint32_t> necklace(length + 1, 0);
  std::vector<uint64_t> start_indices_in_mat;    // Collect start indices
  gen_necklace<T>(1, 1, k, length, necklace, start_indices_in_mat);

  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  for (uint64_t i = 0; i < start_indices_in_mat.size(); i += max_nof_operations) {
    uint64_t nof_operations = std::min((uint64_t)max_nof_operations, start_indices_in_mat.size() - i);
    for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
      VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
      task_p->send_replace_elements_task(
        REPLACE_ELEMENTS,
        config.columns_batch? mat_in + idx_in_batch : mat_in + idx_in_batch * total_elements_one_mat,
        nof_operations,
        start_indices_in_mat,
        i,
        log_nof_rows,
        log_nof_cols,
        config.columns_batch? config.batch_size : 1,
        config.columns_batch? mat_out + idx_in_batch : mat_out + idx_in_batch * total_elements_one_mat);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}


template <typename T>
eIcicleError cpu_matrix_transpose(
  const Device& device, const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
{
  ICICLE_ASSERT(mat_in && mat_out && nof_rows != 0 && nof_cols != 0) << "Invalid argument";

  // check if the number of rows and columns are powers of 2, if not use the basic transpose
  bool is_power_of_2 = (nof_rows & (nof_rows - 1)) == 0 && (nof_cols & (nof_cols - 1)) == 0;
  bool is_inplace = mat_in == mat_out;
  if (!is_inplace) {
    return(out_of_place_matrix_transpose(device, mat_in, nof_rows, nof_cols, config, mat_out));
  } else if (is_power_of_2) {
    return (matrix_transpose_necklaces<T>(mat_in, nof_rows, nof_cols, config, mat_out));
  } else {
    ICICLE_LOG_ERROR << "Matrix transpose is not supported for inplace non power of 2 rows and columns";
    return eIcicleError::INVALID_ARGUMENT;
  }
}

REGISTER_MATRIX_TRANSPOSE_BACKEND("CPU", cpu_matrix_transpose<scalar_t>);
#ifdef EXT_FIELD
REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND("CPU", cpu_matrix_transpose<extension_t>);
#endif // EXT_FIELD

/*********************************** BIT REVERSE ***********************************/
template <typename T>
eIcicleError
cpu_bit_reverse(const Device& device, const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out)
{
  ICICLE_ASSERT(vec_in && vec_out && size != 0) << "Invalid argument";

  uint32_t logn = static_cast<uint32_t>(std::floor(std::log2(size)));
  ICICLE_ASSERT((1ULL << logn) == size) << "Invalid argument - size is not a power of 2";

  // Perform the bit reverse
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    for (uint64_t i = 0; i < size; i += NOF_OPERATIONS_PER_TASK) {
      VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();

      task_p->send_bit_reverse_task(
        BIT_REVERSE,
        logn,
        i,
        std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size - i),
        config.columns_batch? vec_in + idx_in_batch : vec_in + idx_in_batch*size,
        config.columns_batch? config.batch_size : 1,
        config.columns_batch? vec_out + idx_in_batch: vec_out + idx_in_batch*size);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

REGISTER_BIT_REVERSE_BACKEND("CPU", cpu_bit_reverse<scalar_t>);
#ifdef EXT_FIELD
REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND("CPU", cpu_bit_reverse<extension_t>);
#endif // EXT_FIELD

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
  ICICLE_ASSERT(offset + (size_out-1) * stride < size_in) << "Error: Invalid argument - slice out of bound";

  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    for (uint64_t i = 0; i < size_out; i += NOF_OPERATIONS_PER_TASK) {
      VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
      task_p->send_slice_task(
        SLICE,
        config.columns_batch? stride*config.batch_size : stride,
        config.columns_batch? config.batch_size : 1,
        std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size_out - i),
        config.columns_batch? vec_in + idx_in_batch + (offset + i * stride)*config.batch_size : vec_in + idx_in_batch*size_in + offset + i * stride,
        config.columns_batch? vec_out + idx_in_batch + i*config.batch_size : vec_out + idx_in_batch*size_out + i);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

REGISTER_SLICE_BACKEND("CPU", cpu_slice<scalar_t>);
#ifdef EXT_FIELD
REGISTER_SLICE_EXT_FIELD_BACKEND("CPU", cpu_slice<extension_t>);
#endif // EXT_FIELD

/*********************************** Highest non-zero idx ***********************************/
template <typename T>
eIcicleError cpu_highest_non_zero_idx(
  const Device& device, const T* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx /*OUT*/)
{
  ICICLE_ASSERT(input && out_idx && size !=0) << "Error: Invalid argument";
  uint64_t stride = config.columns_batch? config.batch_size : 1;
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    out_idx[idx_in_batch] = -1; // zero vector is considered '-1' since 0 would be zero in vec[0]
    const T* curr_input = config.columns_batch? input + idx_in_batch : input + idx_in_batch * size; // Pointer to the current vector
    for (int64_t i = size - 1; i >= 0; --i) {
      if (curr_input[i * stride] != T::zero()) {
        out_idx[idx_in_batch] = i;
        break;
      }
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND("CPU", cpu_highest_non_zero_idx<scalar_t>);


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
    const T* curr_coeffs = config.columns_batch? coeffs + idx_in_batch : coeffs + idx_in_batch * coeffs_size;
    T* curr_evals = config.columns_batch? evals + idx_in_batch : evals + idx_in_batch * domain_size;
    for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx) {
      curr_evals[eval_idx * stride] = curr_coeffs[(coeffs_size - 1) * stride];
      for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
        curr_evals[eval_idx * stride] = curr_evals[eval_idx * stride] * domain[eval_idx] + curr_coeffs[coeff_idx * stride];
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

  T lc_r = r[deg_r * stride]; // leading coefficient of r
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
  int64_t numerator_deg,
  const T* denumerator,
  int64_t denumerator_deg,
  uint64_t q_size,
  uint64_t r_size,
  const VecOpsConfig& config,
  T* q_out /*OUT*/,
  T* r_out /*OUT*/)
{
  ICICLE_ASSERT(r_size >= numerator_deg)
    << "polynomial division expects r(x) size to be similar to numerator size and higher than numerator degree(x)";
  ICICLE_ASSERT(q_size >= (numerator_deg - denumerator_deg + 1))
    << "polynomial division expects q(x) size to be at least deg(numerator)-deg(denumerator)+1";

  // ICICLE_CHECK(icicle_copy_async(r_out, numerator, r_size * config.batch_size * sizeof(T), config.stream));
  // copy numerator to r_out // FIXME should it be copied using icicle_copy_async?
  for (uint64_t i = 0; i < (numerator_deg+1)*config.batch_size; ++i) {
    r_out[i] = numerator[i];
  }

  uint32_t stride = config.columns_batch? config.batch_size : 1;
  auto deg_r = std::make_unique<int64_t[]>(config.batch_size);
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    const T* curr_denumerator = config.columns_batch? denumerator + idx_in_batch : denumerator + idx_in_batch * (denumerator_deg+1); // Pointer to the current vector
    T* curr_q_out = config.columns_batch? q_out + idx_in_batch : q_out + idx_in_batch * q_size; // Pointer to the current vector
    T* curr_r_out = config.columns_batch? r_out + idx_in_batch : r_out + idx_in_batch * r_size; // Pointer to the current vector
    // invert largest coeff of b
    const T& lc_b_inv = T::inverse(curr_denumerator[denumerator_deg * stride]);
    deg_r[idx_in_batch] = numerator_deg;
    while (deg_r[idx_in_batch] >= denumerator_deg) {
      // each iteration is removing the largest monomial in r until deg(r)<deg(b)
      school_book_division_step_cpu(curr_r_out, curr_q_out, curr_denumerator, deg_r[idx_in_batch], denumerator_deg, lc_b_inv, stride);
      // compute degree of r
      cpu_highest_non_zero_idx(device, r_out, r_size, config, deg_r.get());
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_POLYNOMIAL_DIVISION("CPU", cpu_poly_divide<scalar_t>);