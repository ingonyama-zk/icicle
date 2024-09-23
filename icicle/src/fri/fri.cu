#include <cuda_runtime.h>

#include "fri.h"

#include "fields/field.cuh"
#include "gpu-utils/error_handler.cuh"
#include "gpu-utils/device_context.cuh"

namespace fri {

    namespace {
        template <typename S, typename E>
        __device__ void ibutterfly(E& v0, E& v1, const S& itwid) {
            E tmp = v0;
            v0 = tmp + v1;
            v1 = (tmp - v1) * itwid;
        }

        template <typename S, typename E>
        __global__ void fold_line_kernel(E* evals, S* domain_xs, E alpha, E* folded_evals, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx % 2 == 0 && idx < n) {
                E f_x = evals[idx]; // even
                E f_x_neg = evals[idx + 1]; // odd
                ibutterfly(f_x, f_x_neg, domain_xs[idx / 2].inverse());
                auto folded_evals_idx = idx / 2;
                folded_evals[folded_evals_idx] = f_x + alpha * f_x_neg;
            }
        }

        template <typename S, typename E>
        __global__ void fold_circle_into_line_kernel(E* eval, S* domain_ys, S alpha, S alpha_sq, E* folded_evals, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx % 2 == 0 && idx < n) {
                E f0_px = eval[idx];
                E f1_px = eval[idx + 1];
                ibutterfly(f0_px, f1_px, domain_ys[idx / 2].inverse());
                E f_prime = f0_px + alpha * f1_px;
                auto folded_evals_idx = idx / 2;
                folded_evals[folded_evals_idx] = folded_evals[folded_evals_idx] * alpha_sq + f_prime;
            }
        }
    }

    template <typename S, typename E>
    cudaError_t fold_line(E* evals, S* domain_xs, E alpha, E* folded_evals, int n, FriConfig cfg) {
        CHK_INIT_IF_RETURN();

        cudaStream_t stream = cfg.ctx.stream;
        // Allocate and move line domain evals to device if necessary
        E* d_evals;
        if (!cfg.are_evals_on_device) {
            auto data_size = sizeof(E) * n;
            CHK_IF_RETURN(cudaMallocAsync(&d_evals, data_size, stream));
            CHK_IF_RETURN(cudaMemcpyAsync(d_evals, evals, data_size, cudaMemcpyHostToDevice, stream));
        } else {
            d_evals = evals;
        }
        
        // Allocate and move domain's elements to device if necessary
        S* d_domain_ys;
        if (!cfg.are_domain_elements_on_device) {
            auto data_size = sizeof(S) * n/2;
            CHK_IF_RETURN(cudaMallocAsync(&d_domain_ys, data_size, stream));
            CHK_IF_RETURN(cudaMemcpyAsync(d_domain_ys, domain_ys, data_size, cudaMemcpyHostToDevice, stream));
        } else {
            d_domain_ys = domain_ys;
        }
        
        // Allocate folded_evals if pointer is not a device pointer
        E* d_folded_evals;
        if (!cfg.are_results_on_device) {
            CHK_IF_RETURN(cudaMallocAsync(&d_folded_evals, sizeof(E) * n/2, stream));
        } else {
            d_out = out;
        }

        int block_size = 256;
        int num_blocks = (n / 2 + block_size - 1) / block_size;
        fold_line_kernel<<<num_blocks, block_size, 0 stream>>>(eval, domain_xs, alpha, folded_evals, n);

        // Move folded_evals back to host if requested
        if (!cfg.are_results_on_device) {
            CHK_IF_RETURN(cudaMemcpyAsync(folded_evals, d_folded_evals, sizeof(E) * n/2, cudaMemcpyDeviceToHost, stream));
        }

        // Sync if stream is default stream
        if (stream == 0) CHK_IF_RETURN(cudaStreamSynchronize(stream));

        return CHK_LAST();
    }

    template <typename S, typename E>
    cudaError_t fold_circle_into_line(E* evals, S* domain_ys, S alpha, E* folded_evals, int n, FriConfig cfg) {
        CHK_INIT_IF_RETURN();

        cudaStream_t stream = cfg.ctx.stream;
        // Allocate and move circle domain evals to device if necessary
        E* d_evals;
        if (!cfg.are_evals_on_device) {
            auto data_size = sizeof(E) * n;
            CHK_IF_RETURN(cudaMallocAsync(&d_evals, data_size, stream));
            CHK_IF_RETURN(cudaMemcpyAsync(d_evals, evals, data_size, cudaMemcpyHostToDevice, stream));
        } else {
            d_evals = evals;
        }
        
        // Allocate and move domain's elements to device if necessary
        S* d_domain_ys;
        if (!cfg.are_domain_elements_on_device) {
            auto data_size = sizeof(S) * n/2;
            CHK_IF_RETURN(cudaMallocAsync(&d_domain_ys, data_size, stream));
            CHK_IF_RETURN(cudaMemcpyAsync(d_domain_ys, domain_ys, data_size, cudaMemcpyHostToDevice, stream));
        } else {
            d_domain_ys = domain_ys;
        }
        
        // Allocate folded_evals if pointer is not a device pointer
        E* d_folded_evals;
        bool folded_evals_on_host = is_host_ptr(folded_evals);
        if (!cfg.are_results_on_device) {
            CHK_IF_RETURN(cudaMallocAsync(&d_folded_evals, sizeof(E) * n/2, stream));
        } else {
            d_out = out;
        }

        S alpha_sq = alpha * alpha;
        int block_size = 256;
        int num_blocks = (n / 2 + block_size - 1) / block_size;
        fold_circle_into_line_kernel<<<num_blocks, block_size, 0 stream>>>(d_evals, d_domain_ys, alpha, alpha_sq, d_folded_evals, n);

        // Move folded_evals back to host if requested
        if (!cfg.are_results_on_device) {
            CHK_IF_RETURN(cudaMemcpyAsync(folded_evals, d_folded_evals, sizeof(E) * n/2, cudaMemcpyDeviceToHost, stream));
        }

        // Sync if stream is default stream
        if (stream == 0) CHK_IF_RETURN(cudaStreamSynchronize(stream));

        return CHK_LAST();
    }
} // namespace fri
