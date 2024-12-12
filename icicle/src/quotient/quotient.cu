#include <cstdint>
#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"
#include "vec_ops/vec_ops.cuh"
#include "fields/stark_fields/m31.cuh"
#include "fields/point.cuh"
#include "quotient/quotient.cuh"

namespace quotient {
    namespace {
        template <typename QF, typename CF>
        __device__ QF mul(QF q, CF c) {
            CF a = CF{q.real, q.im1} * c;
            CF b = CF{q.im2, q.im3} * c;
            return QF{a.real, a.imaginary, b.real, b.imaginary};
        }

        template <typename QF>
        __device__ QF pow(QF val, uint32_t scalar) {
            QF res = QF::one();
            while (scalar) {
                if (scalar & 1 == 1) {
                    res = res * val;
                }
                val = val * val;
                scalar >>= 1;
            }
            return res;
        }
    
        template <typename QF, typename F>
        __device__ QF complex_conjugate(QF point) {
            return QF{point.real, point.im1, F::neg(point.im2), F::neg(point.im3)};
        }

        template <typename QF>
        __device__ QF scalar_mul(QF val, uint32_t scalar) {
            QF res = QF::one();
            while (scalar) {
                if (scalar & 1 == 1) {
                    res = res + val;
                }
                val = val + val;
                scalar >>= 1;
            }
            return res;
        }

        template <typename QP, typename QF, typename F>
        __device__ void complex_conjugate_line_coeffs(QP point, QF value, QF alpha, QF* a_out, QF* b_out, QF* c_out) {
            QF a = complex_conjugate<QF, F>(value) - value; 
            QF c = complex_conjugate<QF, F>(point.y) - point.y;
            QF b = (value * c) - (a * point.y);  

            *a_out = alpha * a;
            *b_out = alpha * b;
            *c_out = alpha * c;
        }

        template <typename QP, typename QF, typename CF, typename P>
        __device__ void denominator_inverse(
            ColumnSampleBatch<QP, QF> *sample_batches,
            uint32_t sample_size,
            const P domain_point,
            CF *flat_denominators) {

            for (unsigned int i = 0; i < sample_size; i++) {
                CF prx = CF{sample_batches[i].point->x.real, sample_batches[i].point->x.im1};
                CF pry = CF{sample_batches[i].point->y.real, sample_batches[i].point->y.im1};
                CF pix = CF{sample_batches[i].point->x.im2, sample_batches[i].point->x.im3};
                CF piy = CF{sample_batches[i].point->y.im2, sample_batches[i].point->y.im3};

                CF first_substraction = CF{prx.real - domain_point.x, prx.imaginary};
                CF second_substraction = CF{pry.real - domain_point.y, pry.imaginary};
                CF result = (first_substraction * piy) - (second_substraction * pix);
                flat_denominators[i] = CF::inverse(result);
            }
        }
    }

    template <typename QP, typename QF>
    std::ostream& operator<<(std::ostream& os, const ColumnSampleBatch<QP, QF>& batch) {
        os << "ColumnSampleBatch {\n";
        os << "  point: " << batch.point << "\n";
        os << "  columns: [";
        for (uint32_t i = 0; i < batch.size; ++i) {
            os << batch.columns[i];
            if (i < batch.size - 1) os << ", ";
        }
        os << "]\n";
        os << "  values: [";
        for (uint32_t i = 0; i < batch.size; ++i) {
            os << batch.values[i];
            if (i < batch.size - 1) os << ", ";
        }
        os << "]\n";
        os << "  size: " << batch.size << "\n";
        os << "}\n";
        return os;
    }

    template <typename QP, typename QF, typename F>
    __global__ void column_line_and_batch_random_coeffs(
        ColumnSampleBatch<QP, QF> *sample_batches,
        uint32_t sample_size,
        QF random_coefficient,
        QF *flattened_line_coeffs,
        uint32_t *line_coeffs_sizes,
        QF *batch_random_coeffs) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if(tid < sample_size) {
            batch_random_coeffs[tid] = pow<QF>(random_coefficient, sample_batches[tid].size); 

            // Calculate Column Line Coeffs
            line_coeffs_sizes[tid] = sample_batches[tid].size;
            size_t sample_batches_offset = 0;
            for (int i = 0; i < tid; ++i) {
                sample_batches_offset += line_coeffs_sizes[i];
            }
            sample_batches_offset *= 3;

            QF alpha = QF::one();

            for(size_t j = 0; j < sample_batches[tid].size; ++j) {
                QF sampled_value = sample_batches[tid].values[j];
                alpha = alpha * random_coefficient; 
                QP point = *sample_batches[tid].point;
                QF value = sampled_value; 

                size_t sampled_offset = sample_batches_offset + (j * 3);
                complex_conjugate_line_coeffs<QP, QF, F>(point, value, alpha, &flattened_line_coeffs[sampled_offset], &flattened_line_coeffs[sampled_offset + 1], &flattened_line_coeffs[sampled_offset + 2]); 
            }
        }
    }

    template <typename QP, typename QF, typename CF, typename F, typename P, typename D>
    __global__ void accumulate_quotients_kernel(
        D domain,
        uint32_t domain_size,
        F *columns,
        uint32_t number_of_columns,
        QF random_coefficient,
        ColumnSampleBatch<QP, QF> *samples,
        uint32_t sample_size,
        QF *flattened_line_coeffs,
        uint32_t *line_coeffs_sizes,
        QF *batch_random_coeffs,
        CF *denominator_inverses,
        QF *result ) {
        int row = threadIdx.x + blockDim.x * blockIdx.x;
        if (row < domain_size) {
            CF *denominator_inverses_local = &denominator_inverses[row * sample_size];
            uint32_t index = __brev(row) >> (32 - domain.lg_size());
            P point = domain.at(index);
            denominator_inverse<QP, QF, CF>(
                samples,
                sample_size,
                point,
                denominator_inverses_local
            );
            QF accumulator = QF::zero();
            for(uint32_t i = 0, offset = 0; i < sample_size; ++i) {
                ColumnSampleBatch<QP, QF> sample_batch = samples[i];
                QF *line_coeffs = &flattened_line_coeffs[offset * 3];
                QF batch_coeff = batch_random_coeffs[i];
                uint32_t line_coeffs_size = line_coeffs_sizes[i];

                QF numerator = QF::zero();

                for(uint32_t j = 0; j < line_coeffs_size; ++j) {
                    QF a = line_coeffs[3 * j];
                    QF b = line_coeffs[3 * j + 1];
                    QF c = line_coeffs[3 * j + 2];

                    uint32_t column_index = samples[i].columns[j];
                    QF linear_term = scalar_mul<QF>(a, point.y.limbs_storage.limbs[0]) + b;
                    QF value = scalar_mul<QF>(c, columns[column_index * domain_size + row].limbs_storage.limbs[0]);

                    numerator = numerator + (value - linear_term);
                }

                accumulator = (accumulator * batch_coeff) + mul<QF, CF>(numerator, denominator_inverses_local[i]);
                offset += line_coeffs_size;
            }
            result[row] = accumulator;
        }
    }

    template <typename QP, typename QF>
    __global__ void set_columns_and_values_pointers(ColumnSampleBatch<QP, QF> *d_samples, uint32_t **d_columns_ptrs, QF **d_values_ptrs, QP **d_point_ptrs, int sample_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < sample_size) {

            d_samples[i].columns = d_columns_ptrs[i];
            d_samples[i].values = d_values_ptrs[i];
            d_samples[i].point = d_point_ptrs[i];
        }
    }

    template <typename QP, typename QF, typename CF, typename F, typename P, typename D>
    cudaError_t accumulate_quotients(
        D &domain,
        F *columns, // 2d number_of_columns * domain_size elements
        uint32_t number_of_columns,
        QF &random_coefficient,
        ColumnSampleBatch<QP, QF> *samples,
        uint32_t sample_size,
        uint32_t flattened_line_coeffs_size,
        QuotientConfig &cfg,
        QF *result
    ) {
        CHK_INIT_IF_RETURN();

        cudaStream_t stream = cfg.ctx.stream;

        uint32_t domain_size = domain.size();
        F *d_columns;
        if (cfg.are_columns_on_device) {
            d_columns = columns;
        }
        else {
            CHK_IF_RETURN(cudaMallocAsync(&d_columns, sizeof(F) * number_of_columns * domain_size, stream));
            CHK_IF_RETURN(
            cudaMemcpyAsync(d_columns, columns, sizeof(F) * number_of_columns * domain_size, cudaMemcpyHostToDevice, stream));
        }
        ColumnSampleBatch<QP, QF> *d_samples;
        uint32_t **d_columns_ptrs;
        QF **d_values_ptrs;
        QP **d_point_ptrs;
        uint32_t **h_columns_ptrs;
        QF **h_values_ptrs;
        QP **h_point_ptrs;
        if (cfg.are_sample_points_on_device) {
            d_samples = samples;
        }
        else {
            CHK_IF_RETURN(cudaMallocAsync(&d_samples, sizeof(ColumnSampleBatch<QP, QF>) * sample_size, stream));
            h_columns_ptrs = new uint32_t*[sample_size];
            h_values_ptrs = new QF*[sample_size];
            h_point_ptrs = new QP*[sample_size];

            for (int i = 0; i < sample_size; ++i) {
                // Allocate device memory for columns and values for each struct
                if (samples[i].size > 0) {
                    cudaMallocAsync(&h_columns_ptrs[i], sizeof(uint32_t) * samples[i].size, stream);
                    cudaMemcpyAsync(h_columns_ptrs[i], samples[i].columns, sizeof(uint32_t) * samples[i].size, cudaMemcpyHostToDevice, stream);

                    cudaMallocAsync(&h_values_ptrs[i], sizeof(QF) * samples[i].size, stream);
                    cudaMemcpyAsync(h_values_ptrs[i], samples[i].values, sizeof(QF) * samples[i].size, cudaMemcpyHostToDevice, stream);
                } else {
                    h_columns_ptrs[i] = nullptr;
                    h_values_ptrs[i] = nullptr;
                }
                cudaMallocAsync(&h_point_ptrs[i], sizeof(QP), stream);
                cudaMemcpyAsync(h_point_ptrs[i], samples[i].point, sizeof(QP), cudaMemcpyHostToDevice, stream);
            }
            // Allocate device memory to store the arrays of pointers for columns and values
            CHK_IF_RETURN(cudaMallocAsync(&d_columns_ptrs, sizeof(uint32_t*) * sample_size, stream));
            CHK_IF_RETURN(cudaMallocAsync(&d_values_ptrs, sizeof(QF*) * sample_size, stream));
            CHK_IF_RETURN(cudaMallocAsync(&d_point_ptrs, sizeof(QP*) * sample_size, stream));

            // Copy the host arrays of pointers to device memory
            CHK_IF_RETURN(cudaMemcpyAsync(d_columns_ptrs, h_columns_ptrs, sizeof(uint32_t*) * sample_size, cudaMemcpyHostToDevice, stream));
            CHK_IF_RETURN(cudaMemcpyAsync(d_values_ptrs, h_values_ptrs, sizeof(QF*) * sample_size, cudaMemcpyHostToDevice, stream));
            CHK_IF_RETURN(cudaMemcpyAsync(d_point_ptrs, h_point_ptrs, sizeof(QP*) * sample_size, cudaMemcpyHostToDevice, stream));

            // Copy the struct array from host to device (with placeholder pointers)
            CHK_IF_RETURN(cudaMemcpyAsync(d_samples, samples, sizeof(ColumnSampleBatch<QP, QF>) * sample_size, cudaMemcpyHostToDevice, stream));

            // Kernel to set the `columns` and `values` pointers in the device struct array
            set_columns_and_values_pointers<QP, QF><<<(sample_size + 255) / 256, 256, 0, stream>>>(d_samples, d_columns_ptrs, d_values_ptrs, d_point_ptrs, sample_size);
        }
        
        QF *d_batch_random_coeffs;
        CHK_IF_RETURN(cudaMallocAsync(&d_batch_random_coeffs, sizeof(QF) * sample_size, stream));

        uint32_t *d_line_coeffs_sizes;
        CHK_IF_RETURN(cudaMallocAsync(&d_line_coeffs_sizes, sizeof(uint32_t) * sample_size, stream));

        QF *d_flattened_line_coeffs;
        CHK_IF_RETURN(cudaMallocAsync(&d_flattened_line_coeffs, sizeof(QF) * flattened_line_coeffs_size, stream));

        int block_dim = sample_size < 512 ? sample_size : 512; 
        int num_blocks = block_dim < 512 ? 1 : (sample_size + block_dim - 1) / block_dim;
        column_line_and_batch_random_coeffs<QP, QF, F><<<num_blocks, block_dim, 0, stream>>>(
            d_samples, 
            sample_size, 
            random_coefficient,
            d_flattened_line_coeffs, 
            d_line_coeffs_sizes,
            d_batch_random_coeffs
        );

        QF *d_result;
        if (cfg.are_results_on_device) {
            d_result = result;
        }
        else {
            CHK_IF_RETURN(cudaMallocAsync(&d_result, sizeof(QF) * domain_size, stream));
        }
        
        CF *d_denominator_inverses;
        CHK_IF_RETURN(cudaMallocAsync(&d_denominator_inverses, sizeof(CF) * sample_size * domain_size, stream));


        block_dim = 512;
        num_blocks = (domain_size + block_dim - 1) / block_dim;
        accumulate_quotients_kernel<QP, QF, CF, F, P, D><<<num_blocks, block_dim, 0, stream>>>(
                domain,
                domain_size,
                d_columns,
                number_of_columns,
                random_coefficient,
                d_samples,
                sample_size,
                d_flattened_line_coeffs,
                d_line_coeffs_sizes,
                d_batch_random_coeffs,
                d_denominator_inverses,
                d_result
        );

        if (!cfg.are_results_on_device) {
            CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, sizeof(QF) * domain_size, cudaMemcpyDeviceToHost, stream));
            CHK_IF_RETURN(cudaFreeAsync(d_result, stream));
        }
        CHK_IF_RETURN(cudaFreeAsync(d_denominator_inverses, stream));
        CHK_IF_RETURN(cudaFreeAsync(d_flattened_line_coeffs, stream));
        CHK_IF_RETURN(cudaFreeAsync(d_line_coeffs_sizes, stream));
        CHK_IF_RETURN(cudaFreeAsync(d_batch_random_coeffs, stream));

        if (!cfg.are_sample_points_on_device) {
            for (int i = 0; i < sample_size; ++i) {
                CHK_IF_RETURN(cudaFreeAsync(h_columns_ptrs[i], stream));
                CHK_IF_RETURN(cudaFreeAsync(h_values_ptrs[i], stream));
                CHK_IF_RETURN(cudaFreeAsync(h_point_ptrs[i], stream));
            }
            CHK_IF_RETURN(cudaFreeAsync(d_columns_ptrs, stream));
            CHK_IF_RETURN(cudaFreeAsync(d_point_ptrs, stream));
            CHK_IF_RETURN(cudaFreeAsync(d_values_ptrs, stream));
            CHK_IF_RETURN(cudaFreeAsync(d_samples, stream));
            delete[] h_columns_ptrs;
            delete[] h_values_ptrs;
            delete[] h_point_ptrs;
        }

        if (!cfg.are_columns_on_device) {
            CHK_IF_RETURN(cudaFreeAsync(d_columns, stream));
        }

        if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(stream));
        
        return CHK_LAST();
    }
} // namespace quotient