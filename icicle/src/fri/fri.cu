#include <cuda_runtime.h>

#include "fri/fri.cuh"

#include "fields/field.cuh"
#include "gpu-utils/error_handler.cuh"
#include "gpu-utils/device_context.cuh"

namespace fri {

  namespace {
    template <typename S, typename E>
    __device__ void ibutterfly(E& v0, E& v1, const S& itwid)
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
        S x_domain = domain_xs[idx / 2];
        ibutterfly(f_x, f_x_neg, S::inverse(x_domain));
        auto folded_eval_idx = idx / 2;
        folded_eval[folded_eval_idx] = f_x + alpha * f_x_neg;
      }
    }

    //__device__ int lock = 0;

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

        // __syncthreads();
        // if (idx == 0) {
        //   while (atomicCAS(&lock, 0, 1) != 0)
        //     ;
        //   printf("\n  input: ");
        //   for (int i = 0; i < n; i++) {
        //     auto ival = (const uint32_t*)eval[i];
        //     printf(" 0x%08x%08x%08x%08x, ", ival[0], ival[1], ival[2], ival[3]);
        //   }
        //   printf("\n  output:");

        //   for (int i = 0; i < n / 2; i++) {
        //     auto ival = (const uint32_t*)folded_eval[i];
        //     printf(" 0x%08x%08x%08x%08x, ", ival[0], ival[1], ival[2], ival[3]);
        //   }
        //   printf("\n  domain:");

        //   for (int i = 0; i < n / 2; i++) {
        //     printf(" 0x%08x, ", domain_ys[i].get_limb());
        //   }
        //   printf("\n ******************************** \n");

        //   // Release lock
        //   atomicExch(&lock, 0);
        // }
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
    auto alpha_ptr = (const uint32_t*)alpha;
    auto alpha_sqr_ptr = (const uint32_t*)alpha_sq;

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
} // namespace fri
