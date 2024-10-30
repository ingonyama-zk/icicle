// #pragma once
// #ifndef QUOTIENT_H
// #define QUOTIENT_H

// #include <cstdint>
// #include <cuda_runtime.h>
// #include "fields/point.cuh"
// #include "gpu-utils/device_context.cuh"
// #include "vec_ops/vec_ops.cuh"

// namespace quotient {
//     template <typename QP, typename QF>
//     struct ColumnSampleBatch {
//         QP point; // quad circle point
//         uint32_t *columns;
//         QF *values; //quad extension field element
//         uint32_t size;
//     };

//     struct QuotientConfig {
//         device_context::DeviceContext ctx;
//         bool are_columns_on_device;
//         bool are_sample_points_on_device;
//         bool are_results_on_device;
//         bool is_async;
//     };
//     template <typename QF, typename F>
//     __device__ QF complex_conjugate(QF point) {
//         return QF{point.real, point.im1, F::neg(point.im2), F::neg(point.im3)};
//     }

//     template <typename QF>
//     HOST_DEVICE_INLINE QF pow(QF val, uint32_t scalar) {
//         QF res = QF::one();
//         while (scalar) {
//             if (scalar & 1 == 1) {
//                 res = res * val;
//             }
//             val = val * val;
//             scalar >>= 1;
//         }
//         return res;
//     }

//     template <typename QP, typename QF, typename CF, typename F, typename P, typename C>
//     HOST_DEVICE_INLINE void complex_conjugate_line_coeffs(QP point, QF value, QF alpha, QF* a_out, QF* b_out, QF* c_out) {
//         QF a = complex_conjugate<QF, F>(value) - value; 
//         QF c = complex_conjugate<QF, F>(point.y) - point.y;
//         QF b = (value * c) - (a * point.y);  

//         *a_out = alpha * a;
//         *b_out = alpha * b;
//         *c_out = alpha * c;
//     }

//     template <typename QP, typename QF, typename CF, typename F, typename P, typename C>
//     __global__ void column_line_and_batch_random_coeffs(
//         ColumnSampleBatch<QP, QF> *sample_batches,
//         uint32_t sample_size,
//         QF random_coefficient,
//         QF *flattened_line_coeffs,
//         uint32_t *line_coeffs_sizes,
//         QF *batch_random_coeffs
//     ) {
//         int tid = threadIdx.x + blockDim.x * blockIdx.x;
//         if(tid < sample_size) {
//             batch_random_coeffs[tid] = pow(random_coefficient, sample_batches[tid].size); 

//             // Calculate Column Line Coeffs
//             line_coeffs_sizes[tid] = sample_batches[tid].size;
//             size_t sample_batches_offset = tid * line_coeffs_sizes[tid] * 3; 

//             QF alpha = QF::one();

//             for(size_t j = 0; j < sample_batches[tid].size; ++j) {
//                 QF sampled_value = sample_batches[tid].values[j];
//                 alpha = alpha * random_coefficient; 
//                 QP point = sample_batches[tid].point;
//                 QF value = sampled_value; 

//                 size_t sampled_offset = sample_batches_offset + (j * 3);
//                 complex_conjugate_line_coeffs(point, value, alpha, &flattened_line_coeffs[sampled_offset], &flattened_line_coeffs[sampled_offset + 1], &flattened_line_coeffs[sampled_offset + 2]); 
//             }
//         }
//     }

//     template <typename QP, typename QF, typename CF, typename F, typename P, typename C>
//     __device__ void denominator_inverse(
//         ColumnSampleBatch<QP, QF> *sample_batches,
//         uint32_t sample_size,
//         const P domain_point,
//         CF *flat_denominators) {

//         for (unsigned int i = 0; i < sample_size; i++) {
//             CF prx = CF{sample_batches[i].point.x.real, sample_batches[i].point.x.real.im1};
//             CF pry = CF{sample_batches[i].point.y.real, sample_batches[i].point.y.real.im1};
//             CF pix = CF{sample_batches[i].point.x.im2, sample_batches[i].point.x.real.im3};
//             CF piy = CF{sample_batches[i].point.y.im2, sample_batches[i].point.y.real.im3};

//             CF first_substraction = CF{prx.real - domain_point.x, prx.imaginary};
//             CF second_substraction = CF{pry.real - domain_point.y, pry.imaginary};
//             CF result = (first_substraction * piy) - (second_substraction * pix);
//             flat_denominators[i] = inverse(result);
//         }
//     }

//     template <typename QP, typename QF, typename CF, typename F, typename P, typename C>
//     __global__ void accumulate_quotients_kernel(
//         C half_coset,
//         uint32_t domain_size,
//         F *columns,
//         uint32_t number_of_columns,
//         QF random_coefficient,
//         ColumnSampleBatch<QP, QF> *samples,
//         uint32_t sample_size,
//         QF *flattened_line_coeffs,
//         uint32_t *line_coeffs_sizes,
//         QF *batch_random_coeffs,
//         CF *denominator_inverses,
//         QF *result
//     ) {
//         int row = threadIdx.x + blockDim.x * blockIdx.x;
//         if (row < domain_size) {
//             denominator_inverses = &denominator_inverses[row * sample_size];
//             uint32_t index = __brev(row) >> (32 - half_coset.log_size);
//             P point = P::get_domain_by_index(half_coset.initial_index, half_coset.step_size, domain_size, index);
//             denominator_inverse(
//                 samples,
//                 sample_size,
//                 point,
//                 denominator_inverses
//             );
//             QF accumulator = QF::zero();
//             for(uint32_t i = 0, offset = 0; i < sample_size; ++i) {
//                 ColumnSampleBatch<QP, QF> sample_batch = samples[i];
//                 QF *line_coeffs = &flattened_line_coeffs[offset];
//                 QF batch_coeff = batch_random_coeffs[i];
//                 uint32_t line_coeffs_size = line_coeffs_sizes[i];

//                 QF numerator = QF::zero();

//                 for(uint32_t j = 0; j < line_coeffs_size; j += 3) {
//                     QF a = line_coeffs[j];
//                     QF b = line_coeffs[j + 1];
//                     QF c = line_coeffs[j + 2];

//                     uint32_t column_index = samples[i].columns[j];
//                     QF linear_term = add(mul_by_scalar(a, domain_point.y), b);
//                     QF value = mul_by_scalar(c, columns[column_index * domain_size + row]);

//                     numerator = add(numerator, sub(value, linear_term));
//                 }

//                 accumulator = (accumulator * batch_coeff) + (numerator * denominator_inverses[i]);
//                 offset += (line_coeffs_size * 3);
//             }
//             result[row] = accumulator;
//         }
//     }
//     template <typename QP, typename QF, typename CF, typename F, typename P, typename C>
//     cudaError_t accumulate_quotients(
//         C half_coset,
//         F *columns, // 2d number_of_columns * domain_size elements
//         uint32_t number_of_columns,
//         QF random_coefficient,
//         ColumnSampleBatch<QP, QF> *samples,
//         uint32_t sample_size,
//         uint32_t flattened_line_coeffs_size,
//         QuotientConfig cfg,
//         QF *result
//     ) {
//         CHK_INIT_IF_RETURN();

//         cudaStream_t stream = cfg.ctx.stream;

//         uint32_t coset_size = half.coset.size();
//         F *d_columns;
//         if (cfg.are_columns_on_device) {
//             d_columns = columns;
//         }
//         else {
//             CHK_IF_RETURN(cudaMallocAsync(&d_columns, sizeof(F) * number_of_columns * coset_size, stream));
//             CHK_IF_RETURN(
//             cudaMemcpyAsync(d_columns, columns, sizeof(F) * number_of_columns * coset_size, cudaMemcpyHostToDevice, stream));
//         }
//         ColumnSampleBatch<QP, QF> *d_samples;
//         if (cfg.are_sample_points_on_device) {
//             d_samples = samples;
//         }
//         else {
//             CHK_IF_RETURN(cudaMallocAsync(&d_samples, sizeof(ColumnSampleBatch<QP, QF>) * sample_size, stream));
//             CHK_IF_RETURN(cudaMemcpyAsync(d_samples, samples, sizeof(ColumnSampleBatch<QP, QF>) * sample_size, cudaMemcpyHostToDevice, stream));
//             for (int i = 0; i < sample_size; ++i) {
//                 CHK_IF_RETURN(cudaMallocAsync(&d_samples[i].columns, sizeof(uin32_t) * samples[i].size, stream));
//                 CHK_IF_RETURN(cudaMemcpyAsync(d_samples[i].columns, samples[i].columns, sizeof(uin32_t) * samples[i].size, cudaMemcpyHostToDevice, stream));
//                 CHK_IF_RETURN(cudaMallocAsync(&d_samples[i].values, sizeof(QF) * samples[i].size, stream));
//                 CHK_IF_RETURN(cudaMemcpyAsync(d_samples[i].values, samples[i].values, sizeof(QF) * samples[i].size, cudaMemcpyHostToDevice, stream));
//             }
//         }
        
//         QF *d_batch_random_coeffs;
//         CHK_IF_RETURN(cudaMallocAsync(&d_batch_random_coeffs, sizeof(QF) * sample_size, stream));

//         uint32_t *d_line_coeffs_sizes;
//         CHK_IF_RETURN(cudaMallocAsync(&d_line_coeffs_sizes, sizeof(uint32_t) * sample_size, stream));

//         QF *d_flattened_line_coeffs;
//         CHK_IF_RETURN(cudaMallocAsync(&d_flattened_line_coeffs, sizeof(QF) * flattened_line_coeffs_size, stream));

//         int block_dim = sample_size < 1024 ? sample_size : 1024; 
//         int num_blocks = block_dim < 1024 ? 1 : (sample_size + block_dim - 1) / block_dim;
//         column_line_and_batch_random_coeffs<<<num_blocks, block_dim, 0, stream>>>(
//             d_samples, 
//             sample_size, 
//             random_coefficient,
//             d_flattened_line_coeffs, 
//             d_line_coeffs_sizes,
//             d_batch_random_coeffs
//         );

//         QF *d_result;
//         if (cfg.are_results_on_device) {
//             d_result = result;
//         }
//         else {
//             CHK_IF_RETURN(cudaMallocAsync(&d_result, sizeof(QF) * coset_size, stream));
//         }
        
//         CF *denominator_inverses;
//         CHK_IF_RETURN(cudaMallocAsync(&denominator_inverses, sizeof(CF) * sample_size * coset_size, stream));

//         block_dim = 512;
//         num_blocks = (domain_size + block_dim - 1) / block_dim;
//         accumulate_quotients_kernel<<<num_blocks, block_dim, 0, stream>>>(
//                 half_coset,
//                 coset_size,
//                 columns,
//                 number_of_columns,
//                 random_coefficient,
//                 samples,
//                 sample_size,
//                 d_flattened_line_coeffs,
//                 d_line_coeffs_sizes,
//                 d_batch_random_coeffs,
//                 denominator_inverses,
//                 d_result
//         );

//         if (!cfg.are_results_on_device) {
//             CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, sizeof(QF) * sample_size, cudaMemcpyDeviceToHost, stream));
//             CHK_IF_RETURN(cudaFreeAsync(d_result, stream));
//         }
//         CHK_IF_RETURN(cudaFreeAsync(denominator_inverses, stream));
//         CHK_IF_RETURN(cudaFreeAsync(d_flattened_line_coeffs, stream));
//         CHK_IF_RETURN(cudaFreeAsync(d_line_coeffs_sizes, stream));
//         CHK_IF_RETURN(cudaFreeAsync(d_batch_random_coeffs, stream));

//         if (!cfg.are_sample_points_on_device) {
//             for (int i = 0; i < sample_size; ++i) {
//                 CHK_IF_RETURN(cudaFreeAsync(d_samples[i].columns, stream));
//                 CHK_IF_RETURN(cudaFreeAsync(d_samples[i].values, stream));
//             }
//             CHK_IF_RETURN(cudaFreeAsync(d_samples, stream));
//         }

//         if (!cfg.are_columns_on_device) {
//             CHK_IF_RETURN(cudaFreeAsync(d_columns, stream));
//         }

//         if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(stream));
//     }
// }
// #endif

#pragma once
#ifndef QUOTIENT_H
#define QUOTIENT_H

#include <cstdint>
#include <cuda_runtime.h>
// #include "fields/point.cuh"
#include "gpu-utils/device_context.cuh"
#include "vec_ops/vec_ops.cuh"
#include "fields/stark_fields/m31.cuh"

namespace quotient {
    typedef m31::q_extension_t QF;
    typedef m31::extension_t CF;
    typedef m31::scalar_t F;
    typedef m31::point_t P;
    typedef m31::secure_point_t QP;
    typedef m31::coset_t C;

    HOST_DEVICE_INLINE QF mul(QF q, CF c) {
        CF a = CF{q.real, q.im1} * c;
        CF b = CF{q.im2, q.im3} * c;
        return QF{a.real, a.imaginary, b.real, b.imaginary};
    }

    struct ColumnSampleBatch {
        QP point; // quad circle point
        uint32_t *columns;
        QF *values; //quad extension field element
        uint32_t size;
    };

    struct QuotientConfig {
        device_context::DeviceContext ctx;
        bool are_columns_on_device;
        bool are_sample_points_on_device;
        bool are_results_on_device;
        bool is_async;
    };
    __device__ QF complex_conjugate(QF point) {
        return QF{point.real, point.im1, F::neg(point.im2), F::neg(point.im3)};
    }

    HOST_DEVICE_INLINE QF pow(QF val, uint32_t scalar) {
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

    HOST_DEVICE_INLINE QF scalar_mul(QF val, uint32_t scalar) {
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

    HOST_DEVICE_INLINE void complex_conjugate_line_coeffs(QP point, QF value, QF alpha, QF* a_out, QF* b_out, QF* c_out) {
        QF a = complex_conjugate(value) - value; 
        QF c = complex_conjugate(point.y) - point.y;
        QF b = (value * c) - (a * point.y);  

        *a_out = alpha * a;
        *b_out = alpha * b;
        *c_out = alpha * c;
    }

    __global__ void column_line_and_batch_random_coeffs(
        ColumnSampleBatch *sample_batches,
        uint32_t sample_size,
        QF random_coefficient,
        QF *flattened_line_coeffs,
        uint32_t *line_coeffs_sizes,
        QF *batch_random_coeffs
    ) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if(tid < sample_size) {
            batch_random_coeffs[tid] = pow(random_coefficient, sample_batches[tid].size); 

            // Calculate Column Line Coeffs
            line_coeffs_sizes[tid] = sample_batches[tid].size;
            size_t sample_batches_offset = tid * line_coeffs_sizes[tid] * 3; 

            QF alpha = QF::one();

            for(size_t j = 0; j < sample_batches[tid].size; ++j) {
                QF sampled_value = sample_batches[tid].values[j];
                alpha = alpha * random_coefficient; 
                QP point = sample_batches[tid].point;
                QF value = sampled_value; 

                size_t sampled_offset = sample_batches_offset + (j * 3);
                complex_conjugate_line_coeffs(point, value, alpha, &flattened_line_coeffs[sampled_offset], &flattened_line_coeffs[sampled_offset + 1], &flattened_line_coeffs[sampled_offset + 2]); 
            }
        }
    }

    __device__ void denominator_inverse(
        ColumnSampleBatch *sample_batches,
        uint32_t sample_size,
        const P domain_point,
        CF *flat_denominators) {

        for (unsigned int i = 0; i < sample_size; i++) {
            CF prx = CF{sample_batches[i].point.x.real, sample_batches[i].point.x.im1};
            CF pry = CF{sample_batches[i].point.y.real, sample_batches[i].point.y.im1};
            CF pix = CF{sample_batches[i].point.x.im2, sample_batches[i].point.x.im3};
            CF piy = CF{sample_batches[i].point.y.im2, sample_batches[i].point.y.im3};

            CF first_substraction = CF{prx.real - domain_point.x, prx.imaginary};
            CF second_substraction = CF{pry.real - domain_point.y, pry.imaginary};
            CF result = (first_substraction * piy) - (second_substraction * pix);
            flat_denominators[i] = CF::inverse(result);
        }
    }

    __global__ void accumulate_quotients_kernel(
        C half_coset,
        uint32_t domain_size,
        F *columns,
        uint32_t number_of_columns,
        QF random_coefficient,
        ColumnSampleBatch *samples,
        uint32_t sample_size,
        QF *flattened_line_coeffs,
        uint32_t *line_coeffs_sizes,
        QF *batch_random_coeffs,
        CF *denominator_inverses,
        QF *result
    ) {
        int row = threadIdx.x + blockDim.x * blockIdx.x;
        if (row < domain_size) {
            denominator_inverses = &denominator_inverses[row * sample_size];
            uint32_t index = __brev(row) >> (32 - half_coset.log_size);
            P point = P::get_domain_by_index(half_coset.initial_index, half_coset.step_size, domain_size, index);
            denominator_inverse(
                samples,
                sample_size,
                point,
                denominator_inverses
            );
            QF accumulator = QF::zero();
            for(uint32_t i = 0, offset = 0; i < sample_size; ++i) {
                ColumnSampleBatch sample_batch = samples[i];
                QF *line_coeffs = &flattened_line_coeffs[offset];
                QF batch_coeff = batch_random_coeffs[i];
                uint32_t line_coeffs_size = line_coeffs_sizes[i];

                QF numerator = QF::zero();

                for(uint32_t j = 0; j < line_coeffs_size; j += 3) {
                    QF a = line_coeffs[j];
                    QF b = line_coeffs[j + 1];
                    QF c = line_coeffs[j + 2];

                    uint32_t column_index = samples[i].columns[j];
                    QF linear_term = scalar_mul(a, point.y.limbs_storage.limbs[0]) + b;
                    QF value = scalar_mul(c, columns[column_index * domain_size + row].limbs_storage.limbs[0]);

                    numerator = numerator + (value - linear_term);
                }

                accumulator = (accumulator * batch_coeff) + mul(numerator, denominator_inverses[i]);
                offset += (line_coeffs_size * 3);
            }
            result[row] = accumulator;
        }
    }

    cudaError_t accumulate_quotients(
        C half_coset,
        F *columns, // 2d number_of_columns * domain_size elements
        uint32_t number_of_columns,
        QF random_coefficient,
        ColumnSampleBatch *samples,
        uint32_t sample_size,
        uint32_t flattened_line_coeffs_size,
        QuotientConfig cfg,
        QF *result
    ) {
        CHK_INIT_IF_RETURN();

        cudaStream_t stream = cfg.ctx.stream;

        uint32_t coset_size = half_coset.size();
        F *d_columns;
        if (cfg.are_columns_on_device) {
            d_columns = columns;
        }
        else {
            CHK_IF_RETURN(cudaMallocAsync(&d_columns, sizeof(F) * number_of_columns * coset_size, stream));
            CHK_IF_RETURN(
            cudaMemcpyAsync(d_columns, columns, sizeof(F) * number_of_columns * coset_size, cudaMemcpyHostToDevice, stream));
        }
        ColumnSampleBatch *d_samples;
        if (cfg.are_sample_points_on_device) {
            d_samples = samples;
        }
        else {
            CHK_IF_RETURN(cudaMallocAsync(&d_samples, sizeof(ColumnSampleBatch) * sample_size, stream));
            CHK_IF_RETURN(cudaMemcpyAsync(d_samples, samples, sizeof(ColumnSampleBatch) * sample_size, cudaMemcpyHostToDevice, stream));
            for (int i = 0; i < sample_size; ++i) {
                CHK_IF_RETURN(cudaMallocAsync(&d_samples[i].columns, sizeof(uint32_t) * samples[i].size, stream));
                CHK_IF_RETURN(cudaMemcpyAsync(d_samples[i].columns, samples[i].columns, sizeof(uint32_t) * samples[i].size, cudaMemcpyHostToDevice, stream));
                CHK_IF_RETURN(cudaMallocAsync(&d_samples[i].values, sizeof(QF) * samples[i].size, stream));
                CHK_IF_RETURN(cudaMemcpyAsync(d_samples[i].values, samples[i].values, sizeof(QF) * samples[i].size, cudaMemcpyHostToDevice, stream));
            }
        }
        
        QF *d_batch_random_coeffs;
        CHK_IF_RETURN(cudaMallocAsync(&d_batch_random_coeffs, sizeof(QF) * sample_size, stream));

        uint32_t *d_line_coeffs_sizes;
        CHK_IF_RETURN(cudaMallocAsync(&d_line_coeffs_sizes, sizeof(uint32_t) * sample_size, stream));

        QF *d_flattened_line_coeffs;
        CHK_IF_RETURN(cudaMallocAsync(&d_flattened_line_coeffs, sizeof(QF) * flattened_line_coeffs_size, stream));

        int block_dim = sample_size < 1024 ? sample_size : 1024; 
        int num_blocks = block_dim < 1024 ? 1 : (sample_size + block_dim - 1) / block_dim;
        column_line_and_batch_random_coeffs<<<num_blocks, block_dim, 0, stream>>>(
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
            CHK_IF_RETURN(cudaMallocAsync(&d_result, sizeof(QF) * coset_size, stream));
        }
        
        CF *denominator_inverses;
        CHK_IF_RETURN(cudaMallocAsync(&denominator_inverses, sizeof(CF) * sample_size * coset_size, stream));

        block_dim = 512;
        num_blocks = (coset_size + block_dim - 1) / block_dim;
        accumulate_quotients_kernel<<<num_blocks, block_dim, 0, stream>>>(
                half_coset,
                coset_size,
                columns,
                number_of_columns,
                random_coefficient,
                samples,
                sample_size,
                d_flattened_line_coeffs,
                d_line_coeffs_sizes,
                d_batch_random_coeffs,
                denominator_inverses,
                d_result
        );

        if (!cfg.are_results_on_device) {
            CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, sizeof(QF) * sample_size, cudaMemcpyDeviceToHost, stream));
            CHK_IF_RETURN(cudaFreeAsync(d_result, stream));
        }
        CHK_IF_RETURN(cudaFreeAsync(denominator_inverses, stream));
        CHK_IF_RETURN(cudaFreeAsync(d_flattened_line_coeffs, stream));
        CHK_IF_RETURN(cudaFreeAsync(d_line_coeffs_sizes, stream));
        CHK_IF_RETURN(cudaFreeAsync(d_batch_random_coeffs, stream));

        if (!cfg.are_sample_points_on_device) {
            for (int i = 0; i < sample_size; ++i) {
                CHK_IF_RETURN(cudaFreeAsync(d_samples[i].columns, stream));
                CHK_IF_RETURN(cudaFreeAsync(d_samples[i].values, stream));
            }
            CHK_IF_RETURN(cudaFreeAsync(d_samples, stream));
        }

        if (!cfg.are_columns_on_device) {
            CHK_IF_RETURN(cudaFreeAsync(d_columns, stream));
        }

        if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(stream));
        
        return CHK_LAST();
    }
}
#endif