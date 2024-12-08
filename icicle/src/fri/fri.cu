#include <cuda_runtime.h>

#include "fri/fri.cuh"

#include "fields/field.cuh"
#include "gpu-utils/error_handler.cuh"
#include "gpu-utils/device_context.cuh"

namespace fri {

  namespace {
    template <typename S, typename E>
    DEVICE_INLINE void ibutterfly(E& v0, E& v1, const S& itwid)
    {
      E tmp = v0;
      v0 = tmp + v1;
      v1 = (tmp - v1) * itwid;
    }

    template <typename S, typename E>
    __global__ void fold_line_kernel(E* eval, S* domain_xs, E alpha, E* folded_eval, uint64_t n)
    {
      unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx % 2 == 0 && idx < n) {
        E f_x = eval[idx];         // even
        E f_x_neg = eval[idx + 1]; // odd
        auto folded_eval_idx = idx / 2;
        ibutterfly(f_x, f_x_neg, domain_xs[folded_eval_idx]);
        folded_eval[folded_eval_idx] = f_x + alpha * f_x_neg;
      }
    }

    DEVICE_INLINE uint64_t bit_reverse_index(uint64_t index, uint32_t log_size) {
      if (log_size == 0) {
        return index;
      }
      return __brevll(index) >> ((sizeof(uint64_t) << 3) - log_size);
    }

    const uint32_t FOLD_STEP = 1; // can be changed in future stwo updates

    template <typename S, typename E, typename D>
    __global__ void fold_line_kernel_new(E* eval, D line_domain, E alpha, E* folded_eval, uint64_t n)
    {
      uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx % 2 == 0 && idx < n) {
        E f_x = eval[idx];         // even
        E f_x_neg = eval[idx + 1]; // odd
        auto folded_eval_idx = idx / 2;

        S domain_x = S::inverse(line_domain.at(bit_reverse_index(folded_eval_idx << FOLD_STEP, line_domain.lg_size())));
        ibutterfly(f_x, f_x_neg, domain_x);
        folded_eval[folded_eval_idx] = f_x + alpha * f_x_neg;
      }
    }

    template <typename S, typename E>
    __global__ void fold_circle_into_line_kernel(E* eval, S* domain_ys, E alpha, E alpha_sq, E* folded_eval, uint64_t n)
    {
      unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx % 2 == 0 && idx < n) {
        E f0_px = eval[idx];
        E f1_px = eval[idx + 1];
        auto folded_eval_idx = idx / 2;
        ibutterfly(f0_px, f1_px, domain_ys[folded_eval_idx]);
        E f_prime = f0_px + alpha * f1_px;
        folded_eval[folded_eval_idx] = folded_eval[folded_eval_idx] * alpha_sq + f_prime;
      }
    }

    template <typename S, typename E, typename D>
    __global__ void fold_circle_into_line_kernel_new(E* eval, D domain, E alpha, E alpha_sq, E* folded_eval, uint64_t n)
    {
      unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx % 2 == 0 && idx < n) {
        E f0_px = eval[idx];
        E f1_px = eval[idx + 1];
        auto folded_eval_idx = idx / 2;
        S domain_y = S::inverse(domain.at(bit_reverse_index(folded_eval_idx << FOLD_STEP, domain.lg_size())).y);
        ibutterfly(f0_px, f1_px, domain_y);
        E f_prime = f0_px + alpha * f1_px;
        folded_eval[folded_eval_idx] = folded_eval[folded_eval_idx] * alpha_sq + f_prime;
      }
    }
  } // namespace

  template <typename S, typename E>
  cudaError_t fold_line(E* eval, S* domain_xs, E alpha, E* folded_eval, uint64_t n, FriConfig& cfg)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t& stream = cfg.ctx.stream;
    // Allocate and move line domain evals to device if necessary
    E* d_eval;
    if (!cfg.are_evals_on_device) {
      auto data_size = sizeof(E) * n;
      CHK_IF_RETURN(cudaMallocAsync(&d_eval, data_size, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_eval, eval, data_size, cudaMemcpyHostToDevice, stream));
    } else {
      d_eval = eval;
    }

    // Allocate and move domain's elements to device if necessary
    S* d_domain_xs;
    if (!cfg.are_domain_elements_on_device) {
      auto data_size = sizeof(S) * n / 2;
      CHK_IF_RETURN(cudaMallocAsync(&d_domain_xs, data_size, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_domain_xs, domain_xs, data_size, cudaMemcpyHostToDevice, stream));
    } else {
      d_domain_xs = domain_xs;
    }

    // Allocate folded_eval if pointer is not a device pointer
    E* d_folded_eval;
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_folded_eval, sizeof(E) * n / 2, stream));
    } else {
      d_folded_eval = folded_eval;
    }

    uint64_t num_threads = max(1, min(unsigned(n), 256));
    uint64_t num_blocks = (n + num_threads - 1) / num_threads;
    fold_line_kernel<<<num_blocks, num_threads, 0, stream>>>(d_eval, d_domain_xs, alpha, d_folded_eval, n);

    // Move folded_eval back to host if requested
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(folded_eval, d_folded_eval, sizeof(E) * n / 2, cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(d_folded_eval, stream));
    }
    if (!cfg.are_domain_elements_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_domain_xs, stream)); }
    if (!cfg.are_evals_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_eval, stream)); }

    // Sync if stream is default stream
    if (stream == 0) CHK_IF_RETURN(cudaStreamSynchronize(stream));

    return CHK_LAST();
  }

  template <typename S, typename E, typename D>
  cudaError_t fold_line_new(E* eval, D& line_domain, E alpha, E* folded_eval, uint64_t n, FriConfig& cfg)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = cfg.ctx.stream;
    // Allocate and move line domain evals to device if necessary
    E* d_eval;
    if (!cfg.are_evals_on_device) {
      auto data_size = sizeof(E) * n;
      CHK_IF_RETURN(cudaMallocAsync(&d_eval, data_size, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_eval, eval, data_size, cudaMemcpyHostToDevice, stream));
    } else {
      d_eval = eval;
    }

    // Allocate folded_eval if pointer is not a device pointer
    E* d_folded_eval;
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_folded_eval, sizeof(E) * n / 2, stream));
    } else {
      d_folded_eval = folded_eval;
    }

    uint64_t num_threads = max(1, min(unsigned(n), 256));
    uint64_t num_blocks = (n + num_threads - 1) / num_threads;
    fold_line_kernel_new<S, E, D><<<num_blocks, num_threads, 0, stream>>>(d_eval, line_domain, alpha, d_folded_eval, n);

    // Move folded_eval back to host if requested
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(folded_eval, d_folded_eval, sizeof(E) * n / 2, cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(d_folded_eval, stream));
    }

    if (!cfg.are_evals_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_eval, stream)); }

    // Sync if stream is default stream
    if (stream == 0) CHK_IF_RETURN(cudaStreamSynchronize(stream));

    return CHK_LAST();
  }

  template <typename S, typename E>
  cudaError_t fold_circle_into_line(E* eval, S* domain_ys, E alpha, E* folded_eval, uint64_t n, FriConfig& cfg)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t& stream = cfg.ctx.stream;
    // Allocate and move circle domain evals to device if necessary
    E* d_eval;
    if (!cfg.are_evals_on_device) {
      auto data_size = sizeof(E) * n;
      CHK_IF_RETURN(cudaMallocAsync(&d_eval, data_size, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_eval, eval, data_size, cudaMemcpyHostToDevice, stream));
    } else {
      d_eval = eval;
    }

    // Allocate and move domain's elements to device if necessary
    S* d_domain_ys;
    if (!cfg.are_domain_elements_on_device) {
      auto data_size = sizeof(S) * n / 2;
      CHK_IF_RETURN(cudaMallocAsync(&d_domain_ys, data_size, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_domain_ys, domain_ys, data_size, cudaMemcpyHostToDevice, stream));
    } else {
      d_domain_ys = domain_ys;
    }

    // Allocate folded_evals if pointer is not a device pointer
    E* d_folded_eval;
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_folded_eval, sizeof(E) * n / 2, stream));
    } else {
      d_folded_eval = folded_eval;
    }

    E alpha_sq = alpha * alpha;

    uint64_t num_threads = max(1, min(unsigned(n), 256));
    uint64_t num_blocks = (n + num_threads - 1) / num_threads;
    fold_circle_into_line_kernel<<<num_blocks, num_threads, 0, stream>>>(
      d_eval, d_domain_ys, alpha, alpha_sq, d_folded_eval, n);

    // Move folded_evals back to host if requested
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(folded_eval, d_folded_eval, sizeof(E) * n / 2, cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(d_folded_eval, stream));
    }
    if (!cfg.are_domain_elements_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_domain_ys, stream)); }
    if (!cfg.are_evals_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_eval, stream)); }

    // Sync if stream is default stream
    if (stream == 0) CHK_IF_RETURN(cudaStreamSynchronize(stream));

    return CHK_LAST();
  }

  template <typename S, typename E, typename D>
  cudaError_t fold_circle_into_line_new(E* eval, D& domain, E alpha, E* folded_eval, uint64_t n, FriConfig& cfg)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t& stream = cfg.ctx.stream;
    // Allocate and move circle domain evals to device if necessary
    E* d_eval;
    if (!cfg.are_evals_on_device) {
      auto data_size = sizeof(E) * n;
      CHK_IF_RETURN(cudaMallocAsync(&d_eval, data_size, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_eval, eval, data_size, cudaMemcpyHostToDevice, stream));
    } else {
      d_eval = eval;
    }

    // Allocate folded_evals if pointer is not a device pointer
    E* d_folded_eval;
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_folded_eval, sizeof(E) * n / 2, stream));
    } else {
      d_folded_eval = folded_eval;
    }

    E alpha_sq = alpha * alpha;

    uint64_t num_threads = max(1, min(unsigned(n), 256));
    uint64_t num_blocks = (n + num_threads - 1) / num_threads;
    fold_circle_into_line_kernel_new<S, E, D><<<num_blocks, num_threads, 0, stream>>>(
      d_eval, domain, alpha, alpha_sq, d_folded_eval, n);

    // Move folded_evals back to host if requested
    if (!cfg.are_results_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(folded_eval, d_folded_eval, sizeof(E) * n / 2, cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(d_folded_eval, stream));
    }
    if (!cfg.are_evals_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_eval, stream)); }

    // Sync if stream is default stream
    if (stream == 0) CHK_IF_RETURN(cudaStreamSynchronize(stream));

    return CHK_LAST();
  }
} // namespace fri
