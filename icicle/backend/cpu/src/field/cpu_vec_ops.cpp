
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "tasks_manager.h"

using namespace field_config;
using namespace icicle;

/* Enumeration for the selected operation to execute.
 * The worker task is templated by this enum and based on that the functionality is selected. */
enum VecOperation {
  VECTOR_ADD,
  VECTOR_SUB,
  VECTOR_MUL,
  VECTOR_DIV,
  VECTOR_SUM,
  VECTOR_PRODUCT,
  SCALAR_ADD_VEC,
  SCALAR_SUB_VEC,
  SCALAR_MUL_VEC,
  CONVERT_TO_MONTGOMERY,
  CONVERT_FROM_MONTGOMERY,
  BIT_REVERSE,
  SLICE,

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
  void send_2ops_task(VecOperation operation, const int nof_operations, const T* op_a, const T* op_b, T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_op_b = op_b;
    m_output = output;
    dispatch();
  }

  // Set the operands to execute a task of 1 operand and 1 output and dispatch the task
  void send_1op_task(VecOperation operation, const int nof_operations, const T* op_a, T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_output = output;
    dispatch();
  }
  // Set the operands to execute a task of 1 operand and dispatch the task
  void send_intermidiate_res_task(VecOperation operation, const int nof_operations, const T* op_a)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    dispatch();
  }

  // Set the operands to bitrev operation dispatch the task
  void send_bitrev_task(
    VecOperation operation, int bit_size, uint64_t start_index, const int nof_operations, const T* op_a, T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_output = output;
    m_bit_size = bit_size, m_start_index = start_index;
    dispatch();
  }

  // Set the operands to slice operation dispatch the task
  void send_slice_task(VecOperation operation, uint64_t stride, const int nof_operations, const T* op_a, T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_output = output;
    m_stride = stride;
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
      m_output[i] = m_op_a[i] * T::inverse(m_op_b[i]);
    }
  }
  // Single worker functionality to execute scalar + vector
  void scalar_add_vec()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = *m_op_a + m_op_b[i];
    }
  }
  // Single worker functionality to execute scalar - vector
  void scalar_sub_vec()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = *m_op_a + m_op_b[i];
    }
  }
  // Single worker functionality to execute scalar * vector
  void scalar_mul_vec()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = *m_op_a * m_op_b[i];
    }
  }
  // Single worker functionality to execute sum(vector)
  void vector_sum()
  {
    *m_output = m_op_a[0];
    for (uint64_t i = 1; i < m_nof_operations; ++i) {
      *m_output = *m_output + m_op_a[i];
    }
  }
  // Single worker functionality to execute product(vector)
  void vector_product()
  {
    *m_output = m_op_a[0];
    for (uint64_t i = 1; i < m_nof_operations; ++i) {
      *m_output = *m_output * m_op_a[i];
    }
  }
  // Single worker functionality to execute conversion from barret to montgomery
  // void convert_to_montgomery()
  // {
  //   for (uint64_t i = 0; i < m_nof_operations; ++i) {
  //     m_output[i] = T::to_montgomery(m_op_a[i]);
  //   }
  // }

  // Single worker functionality to execute conversion from montgomery to barret
  // void convert_from_montgomery()
  // {
  //   for (uint64_t i = 0; i < m_nof_operations; ++i) {
  //     m_output[i] = T::from_montgomery(m_op_a[i]);
  //   }
  // }
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
          std::swap(m_output[idx], m_output[rev_idx]);
        }
      } else {                           // out of place calculation
        m_output[idx] = m_op_a[rev_idx]; // set index value
      }
    }
  }

  // Single worker functionality to execute slice
  void slice()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i * m_stride];
    }
  }

  // An array of available function pointers arranged according to the VecOperation enum
  using FunctionPtr = void (VectorOpTask::*)();
  static constexpr std::array<FunctionPtr, static_cast<int>(NOF_OPERATIONS)> functionPtrs = {
    &VectorOpTask::vector_add,              // VECTOR_ADD,
    &VectorOpTask::vector_sub,              // VECTOR_SUB,
    &VectorOpTask::vector_mul,              // VECTOR_MUL,
    &VectorOpTask::vector_div,              // VECTOR_DIV,
    &VectorOpTask::vector_sum,              // VECTOR_SUM
    &VectorOpTask::vector_product,          // VECTOR_PRODUCT
    &VectorOpTask::scalar_add_vec,          // SCALAR_ADD_VEC,
    &VectorOpTask::scalar_sub_vec,          // SCALAR_SUB_VEC,
    &VectorOpTask::scalar_mul_vec,          // SCALAR_MUL_VEC,
    // &VectorOpTask::convert_to_montgomery,   // CONVERT_TO_MONTGOMERY,
    // &VectorOpTask::convert_from_montgomery, // CONVERT_FROM_MONTGOMERY,
    &VectorOpTask::bit_reverse,             // BIT_REVERSE
    &VectorOpTask::slice                    // SLICE
  };

  VecOperation m_operation; // the operation to execute
  int m_nof_operations;     // number of operations to execute for this task
  const T* m_op_a;          // pointer to operand A. Operand A is a vector.
  const T* m_op_b;          // pointer to operand B. Operand B is a vector or scalar
  uint64_t m_start_index;   // index used in bitreverse
  int m_bit_size;           // use in bitrev operation
  uint64_t m_stride;        // used in slice operation
  T* m_output;              // pointer to the output. Can be a vector or scalar pointer
  T m_intermidiate_res;     // pointer to the output. Can be a vector or scalar pointer
};

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
cpu_2vectors_op(VecOperation op, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config));
  for (uint64_t i = 0; i < n; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_2ops_task(op, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, n - i), vec_a + i, vec_b + i, output + i);
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

// Execute a full task from the type vector = scalar (op) vector
template <typename T>
eIcicleError cpu_scalar_vector_op(
  VecOperation op, const T* scalar_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config));
  for (uint64_t i = 0; i < n; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_2ops_task(op, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, n - i), scalar_a, vec_b + i, output + i);
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

///////////////////////////////////////////////////////
// Functions to register at the CPU backend
template <typename T>
eIcicleError
cpu_vector_add(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_ADD, vec_a, vec_b, n, config, output);
}

REGISTER_VECTOR_ADD_BACKEND("CPU", cpu_vector_add<scalar_t>);

/*********************************** ACCUMULATE ***********************************/
template <typename T>
eIcicleError
cpu_vector_accumulate(const Device& device, T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config)
{
  for (uint64_t i = 0; i < n; ++i) {
    vec_a[i] = vec_a[i] + vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_ACCUMULATE_BACKEND("CPU", cpu_vector_accumulate<scalar_t>);

/*********************************** SUB ***********************************/
template <typename T>
eIcicleError
cpu_vector_sub(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_SUB, vec_a, vec_b, n, config, output);
}

REGISTER_VECTOR_SUB_BACKEND("CPU", cpu_vector_sub<scalar_t>);

/*********************************** MUL ***********************************/
template <typename T>
eIcicleError
cpu_vector_mul(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_MUL, vec_a, vec_b, n, config, output);
}

REGISTER_VECTOR_MUL_BACKEND("CPU", cpu_vector_mul<scalar_t>);

/*********************************** DIV ***********************************/
template <typename T>
eIcicleError
cpu_vector_div(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_DIV, vec_a, vec_b, n, config, output);
}

REGISTER_VECTOR_DIV_BACKEND("CPU", cpu_vector_div<scalar_t>);

/*********************************** SUM ***********************************/
template <typename T>
eIcicleError cpu_vector_sum(const Device& device, const T* vec_a, uint64_t n, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config));
  bool output_initialized = false;
  uint64_t vec_s_offset = 0;
  VectorOpTask<T>* task_p;
  // run until all vector deployed and all tasks completed
  do {
    task_p = vec_s_offset < n ? task_manager.get_idle_or_completed_task() : task_manager.get_completed_task();
    if (task_p->is_completed()) {
      *output = output_initialized ? task_p->m_intermidiate_res : *output + task_p->m_intermidiate_res;
    }
    if (vec_s_offset < n) {
      task_p->send_intermidiate_res_task(
        VecOperation::VECTOR_SUM, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, n - vec_s_offset), vec_a + vec_s_offset);
      vec_s_offset += NOF_OPERATIONS_PER_TASK;
    }
  } while (task_p != nullptr);
  return eIcicleError::SUCCESS;
}

// Once backend will support - uncomment the following line
// REGISTER_VECTOR_SUM_BACKEND("CPU", cpu_vector_sum<scalar_t>);
/*********************************** SUM ***********************************/
template <typename T>
eIcicleError cpu_vector_product(const Device& device, const T* vec_a, uint64_t n, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config));
  bool output_initialized = false;
  uint64_t vec_s_offset = 0;
  VectorOpTask<T>* task_p;
  // run until all vector deployed and all tasks completed
  do {
    task_p = vec_s_offset < n ? task_manager.get_idle_or_completed_task() : task_manager.get_completed_task();
    if (task_p->is_completed()) {
      *output = output_initialized ? task_p->m_intermidiate_res : *output * task_p->m_intermidiate_res;
    }
    if (vec_s_offset < n) {
      task_p->send_intermidiate_res_task(
        VecOperation::VECTOR_SUM, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, n - vec_s_offset), vec_a + vec_s_offset);
      vec_s_offset += NOF_OPERATIONS_PER_TASK;
    }
  } while (task_p != nullptr);
  return eIcicleError::SUCCESS;
}

// Once backend will support - uncomment the following line
// REGISTER_VECTOR_SUM_BACKEND("CPU", cpu_vector_sum<scalar_t>);

/*********************************** MUL BY SCALAR***********************************/
template <typename T>
eIcicleError cpu_scalar_mul(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_MUL_VEC, scalar_a, vec_b, n, config, output);
}

REGISTER_SCALAR_MUL_VEC_BACKEND("CPU", cpu_scalar_mul<scalar_t>);

/*********************************** Scalar + Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_add(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_ADD_VEC, scalar_a, vec_b, n, config, output);
}

REGISTER_SCALAR_ADD_VEC_BACKEND("CPU", cpu_scalar_add<scalar_t>);

/*********************************** Scalar - Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_sub(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_SUB_VEC, scalar_a, vec_b, n, config, output);
}

REGISTER_SCALAR_SUB_VEC_BACKEND("CPU", cpu_scalar_sub<scalar_t>);

/*********************************** CONVERT MONTGOMERY ***********************************/
template <typename T>
eIcicleError cpu_convert_montgomery(
  const Device& device, const T* input, uint64_t n, bool is_into, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config));
  for (uint64_t i = 0; i < n; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_1op_task(
      is_into ? CONVERT_TO_MONTGOMERY : CONVERT_FROM_MONTGOMERY, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, n - i),
      input + i, output + i);
  }
  task_manager.wait_done();
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

/*********************************** TRANSPOSE ***********************************/
template <typename T>
eIcicleError cpu_matrix_transpose(
  const Device& device, const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
{
  // Check for invalid arguments
  if (!mat_in || !mat_out || nof_rows == 0 || nof_cols == 0) { return eIcicleError::INVALID_ARGUMENT; }

  // Perform the matrix transpose
  for (uint32_t i = 0; i < nof_rows; ++i) {
    for (uint32_t j = 0; j < nof_cols; ++j) {
      mat_out[j * nof_rows + i] = mat_in[i * nof_cols + j];
    }
  }

  return eIcicleError::SUCCESS;
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
  // Check for invalid arguments
  if (!vec_in || !vec_out || size == 0) { return eIcicleError::INVALID_ARGUMENT; }

  // Calculate log2(size)
  int logn = static_cast<int>(std::floor(std::log2(size)));
  if ((1ULL << logn) != size) {
    return eIcicleError::INVALID_ARGUMENT; // Ensure size is a power of 2
  }

  // Perform the bit reverse
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config));
  for (uint64_t i = 0; i < size; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_bitrev_task(
      BIT_REVERSE, logn, i, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size - i), vec_in, vec_out);
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
  uint64_t size,
  const VecOpsConfig& config,
  T* vec_out)
{
  if (vec_in == nullptr || vec_out == nullptr) {
    ICICLE_LOG_ERROR << "Error: Invalid argument - input or output vector is null";
    return eIcicleError::INVALID_ARGUMENT;
  }

  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config));
  for (uint64_t i = 0; i < size; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_slice_task(
      SLICE, stride, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, size - i), vec_in + offset + i * stride, vec_out + i);
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

REGISTER_SLICE_BACKEND("CPU", cpu_slice<scalar_t>);
#ifdef EXT_FIELD
REGISTER_SLICE_EXT_FIELD_BACKEND("CPU", cpu_slice<extension_t>);
#endif // EXT_FIELD

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
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx) {
    evals[eval_idx] = coeffs[coeffs_size - 1];
    for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
      evals[eval_idx] = evals[eval_idx] * domain[eval_idx] + coeffs[coeff_idx];
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_POLYNOMIAL_EVAL("CPU", cpu_poly_eval<scalar_t>);

/*********************************** Highest non-zero idx ***********************************/
template <typename T>
eIcicleError cpu_highest_non_zero_idx(
  const Device& device, const T* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx /*OUT*/)
{
  *out_idx = -1; // zero vector is considered '-1' since 0 would be zero in vec[0]
  for (int64_t i = size - 1; i >= 0; --i) {
    if (input[i] != T::zero()) {
      *out_idx = i;
      break;
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND("CPU", cpu_highest_non_zero_idx<scalar_t>);

/*============================== polynomial division ==============================*/
template <typename T>
void school_book_division_step_cpu(T* r, T* q, const T* b, int deg_r, int deg_b, const T& lc_b_inv)
{
  int64_t monomial = deg_r - deg_b; // monomial=1 is 'x', monomial=2 is x^2 etc.

  T lc_r = r[deg_r];
  T monomial_coeff = lc_r * lc_b_inv; // lc_r / lc_b

  // adding monomial s to q (q=q+s)
  q[monomial] = monomial_coeff;

  for (int i = monomial; i <= deg_r; ++i) {
    T b_coeff = b[i - monomial];
    r[i] = r[i] - monomial_coeff * b_coeff;
  }
}

template <typename T>
eIcicleError cpu_poly_divide(
  const Device& device,
  const T* numerator,
  int64_t numerator_deg,
  const T* denumerator,
  int64_t denumerator_deg,
  const VecOpsConfig& config,
  T* q_out /*OUT*/,
  uint64_t q_size,
  T* r_out /*OUT*/,
  uint64_t r_size)
{
  ICICLE_ASSERT(r_size >= numerator_deg)
    << "polynomial division expects r(x) size to be similar to numerator size and higher than numerator degree(x)";
  ICICLE_ASSERT(q_size >= (numerator_deg - denumerator_deg + 1))
    << "polynomial division expects q(x) size to be at least deg(numerator)-deg(denumerator)+1";

  ICICLE_CHECK(icicle_copy_async(r_out, numerator, r_size * sizeof(T), config.stream));

  // invert largest coeff of b
  const T& lc_b_inv = T::inverse(denumerator[denumerator_deg]);

  int64_t deg_r = numerator_deg;
  while (deg_r >= denumerator_deg) {
    // each iteration is removing the largest monomial in r until deg(r)<deg(b)
    school_book_division_step_cpu(r_out, q_out, denumerator, deg_r, denumerator_deg, lc_b_inv);

    // compute degree of r
    auto degree_config = default_vec_ops_config();
    cpu_highest_non_zero_idx(device, r_out, deg_r + 1 /*size of R*/, degree_config, &deg_r);
  }

  return eIcicleError::SUCCESS;
}

REGISTER_POLYNOMIAL_DIVISION("CPU", cpu_poly_divide<scalar_t>);