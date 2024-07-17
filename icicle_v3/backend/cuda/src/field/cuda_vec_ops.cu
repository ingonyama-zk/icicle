#include <cuda.h>
#include <stdexcept>

#include "icicle/errors.h"
#include "icicle/vec_ops.h"
#include "gpu-utils/error_handler.h"
#include "error_translation.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

#define MAX_THREADS_PER_BLOCK 256

template <typename E>
E* allocate_on_device(size_t byte_size, cudaStream_t cuda_stream)
{
  E* device_mem = nullptr;
  cudaMallocAsync(&device_mem, byte_size, cuda_stream);
  CHK_LAST();

  return device_mem;
}

template <typename E>
const E* allocate_and_copy_to_device(const E* host_mem, size_t byte_size, cudaStream_t cuda_stream)
{
  E* device_mem = nullptr;
  cudaMallocAsync(&device_mem, byte_size, cuda_stream);
  cudaMemcpyAsync(device_mem, host_mem, byte_size, cudaMemcpyHostToDevice, cuda_stream);
  CHK_LAST();

  return device_mem;
}

template <typename E, void (*Kernel)(const E*, const E*, int, E*)>
cudaError_t vec_op(const E* a, const E* b, int size_a, int size_b, const VecOpsConfig& config, E* result, int size_res)
{
  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  a = config.is_a_on_device ? a : allocate_and_copy_to_device(a, size_a * sizeof(E), cuda_stream);
  b = config.is_b_on_device ? b : allocate_and_copy_to_device(b, size_b * sizeof(E), cuda_stream);
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(size_res * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  int num_threads = MAX_THREADS_PER_BLOCK;
  int num_blocks = (size_res + num_threads - 1) / num_threads;
  Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(a, b, size_res, d_result);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, size_res * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)a, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)b, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

/*============================== add ==============================*/
template <typename E>
__global__ void add_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { result[tid] = element_vec1[tid] + element_vec2[tid]; }
}

template <typename E>
eIcicleError
add_cuda(const Device& device, const E* vec_a, const E* vec_b, int n, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, add_kernel>(vec_a, vec_b, n, n, config, result, n);
  return translateCudaError(err);
}

template <typename E, typename S>
__global__ void add_scalar_kernel(const S* scalar, const E* element_vec, int n, E* result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { result[tid] = element_vec[tid] + (*scalar); }
}

template <typename E>
eIcicleError
add_scalar_cuda(const Device& device, const E* scalar_a, const E* vec_b, int n, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, add_scalar_kernel>(scalar_a, vec_b, 1, n, config, result, n);
  return translateCudaError(err);
}

/*============================== sub ==============================*/
template <typename E>
__global__ void sub_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { result[tid] = element_vec1[tid] - element_vec2[tid]; }
}

template <typename E>
eIcicleError
sub_cuda(const Device& device, const E* vec_a, const E* vec_b, int n, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, sub_kernel>(vec_a, vec_b, n, n, config, result, n);
  return translateCudaError(err);
}

template <typename E, typename S>
__global__ void sub_scalar_kernel(const S* scalar, const E* element_vec, int n, E* result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { result[tid] = element_vec[tid] - (*scalar); }
}

template <typename E>
eIcicleError
sub_scalar_cuda(const Device& device, const E* scalar_a, const E* vec_b, int n, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, sub_scalar_kernel>(scalar_a, vec_b, 1, n, config, result, n);
  return translateCudaError(err);
}

/*============================== mul ==============================*/
template <typename E>
__global__ void mul_kernel(const E* vec_a, const E* vec_b, int n, E* result)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) { result[tid] = vec_a[tid] * vec_b[tid]; }
}

template <typename E>
eIcicleError
mul_cuda(const Device& device, const E* vec_a, const E* vec_b, int n, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, mul_kernel>(vec_a, vec_b, n, n, config, result, n);
  return translateCudaError(err);
}

template <typename E, typename S>
__global__ void mul_scalar_kernel(const S* scalar, const E* element_vec, int n, E* result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { result[tid] = element_vec[tid] * (*scalar); }
}

template <typename E>
eIcicleError
mul_scalar_cuda(const Device& device, const E* scalar_a, const E* vec_b, int n, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, mul_scalar_kernel>(scalar_a, vec_b, 1, n, config, result, n);
  return translateCudaError(err);
}

/*============================== div ==============================*/
template <typename E>
__global__ void div_element_wise_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
{
  // TODO:implement better based on https://eprint.iacr.org/2008/199
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { result[tid] = element_vec1[tid] * E::inverse(element_vec2[tid]); }
}

template <typename E>
eIcicleError
div_cuda(const Device& device, const E* vec_a, const E* vec_b, int n, const VecOpsConfig& config, E* result)
{
  cudaError_t err = vec_op<E, div_element_wise_kernel>(vec_a, vec_b, n, n, config, result, n);
  return translateCudaError(err);
}
/*============================== transpose ==============================*/

template <typename E>
__global__ void transpose_kernel(const E* in, E* out, uint32_t row_size, uint32_t column_size)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= row_size * column_size) return;
  out[(tid % row_size) * column_size + (tid / row_size)] = in[tid];
}

template <typename E>
cudaError_t transpose_matrix(
  const E* mat_in,
  E* mat_out,
  uint32_t row_size,
  uint32_t column_size,
  cudaStream_t stream,
  bool on_device,
  bool is_async)
{
  int number_of_threads = MAX_THREADS_PER_BLOCK;
  int number_of_blocks = (row_size * column_size + number_of_threads - 1) / number_of_threads;

  const E* d_mat_in;
  E* d_allocated_input = nullptr;
  E* d_mat_out;
  if (!on_device) {
    CHK_IF_RETURN(cudaMallocAsync(&d_allocated_input, row_size * column_size * sizeof(E), stream));
    CHK_IF_RETURN(
      cudaMemcpyAsync(d_allocated_input, mat_in, row_size * column_size * sizeof(E), cudaMemcpyHostToDevice, stream));

    CHK_IF_RETURN(cudaMallocAsync(&d_mat_out, row_size * column_size * sizeof(E), stream));
    d_mat_in = d_allocated_input;
  } else {
    d_mat_in = mat_in;
    d_mat_out = mat_out;
  }

  transpose_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(d_mat_in, d_mat_out, row_size, column_size);

  if (!on_device) {
    CHK_IF_RETURN(
      cudaMemcpyAsync(mat_out, d_mat_out, row_size * column_size * sizeof(E), cudaMemcpyDeviceToHost, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_mat_out, stream));
    CHK_IF_RETURN(cudaFreeAsync(d_allocated_input, stream));
  }
  if (!is_async) return CHK_STICKY(cudaStreamSynchronize(stream));

  return CHK_LAST();
}

template <typename F>
eIcicleError matrix_transpose_cuda(
  const Device& device, const F* in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, F* out)
{
  // TODO relax this limitation
  ICICLE_ASSERT(config.is_a_on_device == config.is_result_on_device)
    << "CUDA matrix transpose expects both input and output on host or on device";

  // assert that it is not an inplace computation
  const bool is_on_device = config.is_a_on_device;
  const bool is_inplace = in == out;
  ICICLE_ASSERT(!is_on_device || !is_inplace) << "(CUDA) matrix-transpose-inplace not implemented";
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  auto err = transpose_matrix(in, out, nof_cols, nof_rows, cuda_stream, config.is_a_on_device, config.is_async);
  return translateCudaError(err);
}

/*============================== Bit-reverse ==============================*/

template <typename E>
__global__ void bit_reverse_kernel(const E* input, uint64_t n, unsigned shift, E* output)
{
  uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Handling arbitrary vector size
  if (tid < n) {
    int reversed_index = __brevll(tid) >> shift;
    output[reversed_index] = input[tid];
  }
}
template <typename E>
__global__ void bit_reverse_inplace_kernel(E* input, uint64_t n, unsigned shift)
{
  uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Handling arbitrary vector size
  if (tid < n) {
    int reversed_index = __brevll(tid) >> shift;
    if (reversed_index > tid) {
      E temp = input[tid];
      input[tid] = input[reversed_index];
      input[reversed_index] = temp;
    }
  }
}

template <typename E>
cudaError_t bit_reverse_cuda_impl(const E* input, uint64_t size, const VecOpsConfig& cfg, E* output)
{
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(cfg.stream);

  if (size & (size - 1)) THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, "bit_reverse: size must be a power of 2");
  if ((input == output) & (cfg.is_a_on_device != cfg.is_result_on_device))
    THROW_ICICLE_ERR(
      eIcicleError::INVALID_ARGUMENT, "bit_reverse: equal devices should have same is_on_device parameters");

  E* d_output;
  if (cfg.is_result_on_device) {
    d_output = output;
  } else {
    // allocate output on gpu
    CHK_IF_RETURN(cudaMallocAsync(&d_output, sizeof(E) * size, cuda_stream));
  }

  uint64_t shift = __builtin_clzll(size) + 1;
  uint64_t num_blocks = (size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

  if ((input != output) & cfg.is_a_on_device) {
    bit_reverse_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(input, size, shift, d_output);
  } else {
    if (!cfg.is_a_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(d_output, input, sizeof(E) * size, cudaMemcpyHostToDevice, cuda_stream));
    }
    bit_reverse_inplace_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(d_output, size, shift);
  }
  if (!cfg.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(output, d_output, sizeof(E) * size, cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_output, cuda_stream));
  }
  if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cuda_stream));
  return CHK_LAST();
}

template <typename T>
eIcicleError bit_reverse_cuda(const Device& device, const T* in, uint64_t size, const VecOpsConfig& config, T* out)
{
  auto err = bit_reverse_cuda_impl<T>(in, size, config, out);
  return translateCudaError(err);
}

/*============================== slice ==============================*/
template <typename T>
__global__ void slice_kernel(const T* in, T* out, int offset, int stride, int size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { out[tid] = in[offset + tid * stride]; }
}

template <typename E>
cudaError_t slice(const E* vec_a, int offset, int stride, int size, const VecOpsConfig& config, E* result)
{
  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  // need to copy consecutive memory where output elements reside in vec_a
  const int input_elements_required = stride * size;
  const E* d_vec_a = config.is_a_on_device
                       ? vec_a
                       : allocate_and_copy_to_device(vec_a + offset, input_elements_required * sizeof(E), cuda_stream);
  offset = config.is_a_on_device ? offset : 0; // already copied from offset so reset to zero
  E* d_result = config.is_result_on_device ? result : allocate_on_device<E>(size * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  int num_threads = MAX_THREADS_PER_BLOCK;
  int num_blocks = (input_elements_required + num_threads - 1) / num_threads;
  slice_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(d_vec_a, d_result, offset, stride, size);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_result, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)d_vec_a, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
eIcicleError slice_cuda(
  const Device& device, const E* vec_a, int offset, int stride, int size, const VecOpsConfig& config, E* result)
{
  cudaError_t err = slice<E>(vec_a, offset, stride, size, config, result);
  return translateCudaError(err);
}

/*============================== highest non-zero idx ==============================*/
template <typename T>
__global__ void highest_non_zero_idx_kernel(const T* vec, uint64_t len, int64_t* idx)
{
  *idx = -1; // -1 for all zeros vec
  for (int64_t i = len - 1; i >= 0; --i) {
    if (vec[i] != T::zero()) {
      *idx = i;
      return;
    }
  }
}

template <typename E>
cudaError_t _highest_non_zero_idx(const E* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx)
{
  // TODO: parallelize kernel? Note that when used for computing degree of polynomial, typically the largest coefficient
  // is expected in the higher half since memory is allocate based on #coefficients

  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  // need to copy consecutive memory where output elements reside in vec_a
  input = config.is_a_on_device ? input : allocate_and_copy_to_device(input, size * sizeof(E), cuda_stream);
  int64_t* d_out_idx = config.is_result_on_device ? out_idx : allocate_on_device<int64_t>(sizeof(int64_t), cuda_stream);

  // Call the kernel to perform element-wise operation
  int num_threads = MAX_THREADS_PER_BLOCK;
  int num_blocks = (size + num_threads - 1) / num_threads;
  highest_non_zero_idx_kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, size, d_out_idx);

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(out_idx, d_out_idx, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_out_idx, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)input, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
eIcicleError highest_non_zero_idx_cuda(
  const Device& device, const E* vec_a, uint64_t size, const VecOpsConfig& config, int64_t* out_idx)
{
  cudaError_t err = _highest_non_zero_idx<E>(vec_a, size, config, out_idx);
  return translateCudaError(err);
}

/*============================== polynomial evaluation ==============================*/
// TODO Yuval: implement efficient reduction and support batch evaluation
template <typename T>
__global__ void dummy_reduce(const T* arr, int size, T* output)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > 0) return;

  *output = arr[0];
  for (int i = 1; i < size; ++i) {
    *output = *output + arr[i];
  }
}

template <typename T>
__global__ void evaluate_polynomial_without_reduction(const T* x, const T* coeffs, int num_coeffs, T* tmp)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_coeffs) { tmp[tid] = coeffs[tid] * T::pow(*x, tid); }
}

template <typename E>
cudaError_t _poly_eval(
  const scalar_t* coeffs,
  uint64_t coeffs_size,
  const scalar_t* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  scalar_t* evals /*OUT*/)
{
  // TODO: implement fast version

  CHK_INIT_IF_RETURN();

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

  // allocate device memory and copy if input/output not on device already
  // need to copy consecutive memory where output elements reside in vec_a
  coeffs = config.is_a_on_device ? coeffs : allocate_and_copy_to_device(coeffs, coeffs_size * sizeof(E), cuda_stream);
  domain = config.is_b_on_device ? domain : allocate_and_copy_to_device(domain, domain_size * sizeof(E), cuda_stream);
  scalar_t* d_evals = config.is_result_on_device ? evals : allocate_on_device<E>(domain_size * sizeof(E), cuda_stream);
  scalar_t* d_tmp = allocate_on_device<E>(coeffs_size * sizeof(E), cuda_stream);

  // Call the kernel to perform element-wise operation
  // TODO Yuval: other methods can avoid this allocation. Also for eval_on_domain() no need to reallocate every
  // time
  // TODO Yuval: maybe use Horner's rule and just evaluate each domain point per thread. Alternatively Need to
  // reduce in parallel.

  int num_threads = MAX_THREADS_PER_BLOCK;
  int num_blocks = (coeffs_size + num_threads - 1) / num_threads;
  // TODO Yuval : replace stupid loop with cuda parallelism but need to avoid the d_tmp since it's O(n) for poly of
  // degree n
  for (uint64_t i = 0; i < domain_size; ++i) {
    evaluate_polynomial_without_reduction<<<num_blocks, num_threads, 0, cuda_stream>>>(
      &domain[i], coeffs, coeffs_size, d_tmp); // TODO Yuval: parallelize kernel
    dummy_reduce<<<1, 1, 0, cuda_stream>>>(d_tmp, coeffs_size, &d_evals[i]);
  }

  // copy back result to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(evals, d_evals, domain_size * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_evals, cuda_stream));
  }

  // release device memory, if allocated
  // the cast is ugly but it makes the code more compact
  if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)coeffs, cuda_stream)); }
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)domain, cuda_stream)); }
  CHK_IF_RETURN(cudaFreeAsync(d_tmp, cuda_stream));

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename E>
eIcicleError poly_eval_cuda(
  const Device& device,
  const scalar_t* coeffs,
  uint64_t coeffs_size,
  const scalar_t* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  scalar_t* evals /*OUT*/)
{
  cudaError_t err = _poly_eval<E>(coeffs, coeffs_size, domain, domain_size, config, evals);
  return translateCudaError(err);
}

/*============================== polynomial division ==============================*/

template <typename T>
__global__ void school_book_division_step(T* r, T* q, const T* b, int deg_r, int denumerator_deg, T lc_b_inv)
{
  // computing one step 'r = r-sb' (for 'a = q*b+r') where s is a monomial such that 'r-sb' removes the highest degree
  // of r.
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t monomial = deg_r - denumerator_deg; // monomial=1 is 'x', monomial=2 is x^2 etc.
  if (tid > deg_r) return;

  T lc_r = r[deg_r];
  T monomial_coeff = lc_r * lc_b_inv; // lc_r / lc_b
  if (tid == 0) {
    // adding monomial s to q (q=q+s)
    q[monomial] = monomial_coeff;
  }

  if (tid < monomial) return;

  T b_coeff = b[tid - monomial];
  r[tid] = r[tid] - monomial_coeff * b_coeff;
}

template <typename T>
cudaError_t _poly_divide_cuda(
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
  CHK_INIT_IF_RETURN();

  ICICLE_ASSERT(r_size >= (1 + denumerator_deg))
    << "polynomial division expects r(x) size to be similar to numerator(x)";
  ICICLE_ASSERT(q_size >= (numerator_deg - denumerator_deg + 1))
    << "polynomial division expects q(x) size to be at least deg(numerator)-deg(denumerator)+1";

  auto cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
  // copy denum to device if need to.
  denumerator = config.is_b_on_device
                  ? denumerator
                  : allocate_and_copy_to_device(denumerator, (1 + denumerator_deg) * sizeof(T), cuda_stream);
  T* d_r_out = config.is_result_on_device ? r_out : allocate_on_device<T>(r_size * sizeof(T), cuda_stream);
  T* d_q_out = config.is_result_on_device ? q_out : allocate_on_device<T>(q_size * sizeof(T), cuda_stream);
  // note that no need to copy numerator to device since we copy it to r already
  ICICLE_CHECK(icicle_copy_async(d_r_out, numerator, (1 + numerator_deg) * sizeof(T), config.stream));

  // invert largest coeff of b
  T denum_highest_coeff;
  ICICLE_CHECK(icicle_copy(&denum_highest_coeff, denumerator + denumerator_deg, sizeof(T)));
  const T& lc_b_inv = T::inverse(denum_highest_coeff);

  int64_t deg_r = numerator_deg;
  while (deg_r >= denumerator_deg) {
    // each iteration is removing the largest monomial in r until deg(r)<deg(b)
    const int NOF_THREADS = 128;
    const int NOF_BLOCKS = ((deg_r + 1) + NOF_THREADS - 1) / NOF_THREADS; // 'deg_r+1' is number of elements in R
    school_book_division_step<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
      d_r_out, d_q_out, denumerator, deg_r, denumerator_deg, lc_b_inv);

    // compute degree of r
    auto degree_config = default_vec_ops_config();
    degree_config.is_a_on_device = true;
    degree_config.stream = config.stream;
    degree_config.is_async = config.is_async;
    _highest_non_zero_idx(d_r_out, deg_r + 1 /*size of R*/, degree_config, &deg_r);
  }

  // copy back output to host if need to
  if (!config.is_result_on_device) {
    CHK_IF_RETURN(cudaMemcpyAsync(r_out, d_r_out, r_size * sizeof(T), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaMemcpyAsync(q_out, d_q_out, q_size * sizeof(T), cudaMemcpyDeviceToHost, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_r_out, cuda_stream));
    CHK_IF_RETURN(cudaFreeAsync(d_q_out, cuda_stream));
  }

  // release device memory, if allocated
  if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)denumerator, cuda_stream)); }

  // wait for stream to empty is not async
  if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

  return CHK_LAST();
}

template <typename T>
eIcicleError poly_divide_cuda(
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
  cudaError_t err = _poly_divide_cuda<T>(
    numerator, numerator_deg, denumerator, denumerator_deg, config, q_out /*OUT*/, q_size, r_out /*OUT*/, r_size);
  return translateCudaError(err);
}

/************************************ REGISTRATION ************************************/

REGISTER_VECTOR_ADD_BACKEND("CUDA", add_cuda<scalar_t>);
REGISTER_VECTOR_SUB_BACKEND("CUDA", sub_cuda<scalar_t>);
REGISTER_VECTOR_MUL_BACKEND("CUDA", mul_cuda<scalar_t>);
REGISTER_VECTOR_DIV_BACKEND("CUDA", div_cuda<scalar_t>);
REGISTER_SCALAR_MUL_VEC_BACKEND("CUDA", mul_scalar_cuda<scalar_t>);
REGISTER_SCALAR_ADD_VEC_BACKEND("CUDA", add_scalar_cuda<scalar_t>);
REGISTER_SCALAR_SUB_VEC_BACKEND("CUDA", sub_scalar_cuda<scalar_t>);
REGISTER_MATRIX_TRANSPOSE_BACKEND("CUDA", matrix_transpose_cuda<scalar_t>);
REGISTER_BIT_REVERSE_BACKEND("CUDA", bit_reverse_cuda<scalar_t>);
REGISTER_SLICE_BACKEND("CUDA", slice_cuda<scalar_t>);
REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND("CUDA", highest_non_zero_idx_cuda<scalar_t>)
REGISTER_POLYNOMIAL_EVAL("CUDA", poly_eval_cuda<scalar_t>);
REGISTER_POLYNOMIAL_DIVISION("CUDA", poly_divide_cuda<scalar_t>);

#ifdef EXT_FIELD
REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND("CUDA", add_cuda<extension_t>);
REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND("CUDA", sub_cuda<extension_t>);
REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND("CUDA", mul_cuda<extension_t>);
REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND("CUDA", matrix_transpose_cuda<extension_t>);
REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND("CUDA", bit_reverse_cuda<extension_t>);
REGISTER_SLICE_EXT_FIELD_BACKEND("CUDA", slice_cuda<extension_t>);
#endif // EXT_FIELD
