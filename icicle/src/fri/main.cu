// #include <chrono>
// #include <fstream>
// #include <iostream>
// #include "stwo.cuh"
// #include "fib_eval_t.cuh"

// static constexpr uint32_t BLOCK_SIZE = 256;

// trace[3]
// preprocessing
// execution <---
// interaction

// m31_t execution_trace[row][col];

// __global__ void k_eval_row(
//     const uint32_t total_constraints,
//     const uint32_t m_cols,
//     qm31_t *res,
//     const qm31_t *random_coeff_powers,
//     const m31_t *denom_inv,
//     const m31_t *trace
// ) {
//     const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     const m31_t *row = trace + thread_id * m_cols;
//     fib_eval_t eval(random_coeff_powers, row, total_constraints, m_cols);
//     eval.evaluate();
//     const qm31_t row_res = eval.get_row_res();
//     res[thread_id] = row_res * denom_inv[thread_id >> 20];
// }

// void generate_random_coeff_powers(qm31_t *random_coeff_powers, const qm31_t &x, const uint32_t n) {
//     qm31_t acc = qm31_t::one();
//     for (uint32_t i = 0; i < n; i++) {
//         random_coeff_powers[i] = acc;
//         acc = acc * x;
//     }
// }

// m31_t coset_vanishing(const coset_t &coset, point_t p) {
//     p = p - coset.initial_point + coset.at(coset.step_size >> 1);
//     m31_t x = p.x;
//     for (uint32_t i = 1; i < coset.log_size; i++) {
//         x = point_t::dbl_x(x);
//     }
//     return x;
// }

// static constexpr uint32_t BLOCK_SIZE = 256;

// void evaluate_constraint_quotients_on_domain(
//     const prover_ctx_t &prover,
//     const component_ctx_t &component
// ) {
//     const uint32_t expand_log_size = component.eval_log_size - component.domain_log_size;
//     const uint32_t domain_size = 1 << component.domain_log_size;
//     const uint32_t eval_size = 1 << component.eval_log_size;
//     const uint32_t expand_size = 1 << expand_log_size;
//     const domain_t trace_domain(component.domain_log_size);
//     const domain_t eval_domain(component.eval_log_size);
//     m31_t *denom_inv;
//     cudaMallocManaged(&denom_inv, sizeof(m31_t) * expand_size);

//     // TODO: bit reverse
//     for (int i = 0; i < expand_size; i++) {
//         denom_inv[i] = m31_t::inverse(coset_vanishing(trace_domain.coset, eval_domain.at(i)));
//     }

//     // CPU version
//     // for (int i = 0; i < component.execution_trace.n_rows; i++) {
//     //     const m31_t *row = component.execution_trace.get_row(i);
//     //     fib_eval_t eval(random_coeff_powers, row, total_constraints, component.execution_trace.m_cols);
//     //     eval.evaluate();
//     //     const qm31_t row_res = eval.get_row_res();
//     //     res[i] = row_res * denom_inv[i >> component.domain_log_size];
//     // }

//     // GPU version
//     k_eval_row<<<component.execution_trace.n_rows / BLOCK_SIZE, BLOCK_SIZE>>>(
//         prover.total_constraints,
//         component.execution_trace.m_cols,
//         prover.composition_poly,
//         prover.random_coeff_powers,
//         denom_inv,
//         component.execution_trace.trace
//     );
//     cudaDeviceSynchronize();
// }

// void compute_composition_polynomial(
//     prover_ctx_t &prover,
//     const qm31_t &random_coeff
// ) {
//     generate_random_coeff_powers(prover.random_coeff_powers, random_coeff, prover.total_constraints);

//     for (uint32_t i = 0; i < prover.components.size(); i++) {
//         evaluate_constraint_quotients_on_domain(prover, prover.components[i]);
//     }

//     // for (uint32_t i = 0; i < total_constraints; i++) {
//     //     qm31_t p = prover_ctx.secure_powers[i];
//     //     printf("%x, %x, %x, %x\n", p.real.get_limb(), p.im1.get_limb(), p.im2.get_limb(), p.im3.get_limb());
//     // }
// }

// void readBinaryFile(const std::string &filename, m31_t *buffer, std::size_t numElements) {
//     // Open file in binary mode
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         throw std::runtime_error("Could not open file: " + filename);
//     }

//     // Calculate how many bytes we want to read
//     const std::size_t bytesToRead = numElements * sizeof(uint32_t);

//     // Read the data into the buffer
//     file.read(reinterpret_cast<char *>(buffer), bytesToRead);

//     // Check if the read was successful or partial
//     if (!file) {
//         // file.gcount() tells how many bytes were actually read
//         std::size_t bytesRead = file.gcount();
//         throw std::runtime_error(
//             "Error or partial read. Expected " + std::to_string(bytesToRead) +
//             " bytes, got " + std::to_string(bytesRead) + " bytes."
//         );
//     }
// }

// fib_wide example
// total_constraints: 98
// components: 1
// trace_domain: 1048576 (2^20)
// eval_domain: 2097152 (2^21)
// trace dimensions:
//   preprocessing_trace [col * row]: 0
//   execution_trace [col * row]: 100 * 2097152
//   interaction_trace [col * row]: 0

// int main() {
//     constexpr uint32_t total_constraints = 98;
//     const qm31_t test_random_coeff = {0x583B16E6, 0x496EFFDF, 0xEBD5346, 0x40D4077F};

//     prover_ctx_t p;
//     p.total_constraints = total_constraints;
//     component_ctx_t c;
//     c.domain_log_size = 20;
//     c.eval_log_size = 21;
//     c.allocate_execution_trace(100, 2097152);


//     // load test data
//     readBinaryFile("/home/tonyw/scratch/stwo-playground/eval.bin", c.execution_trace.trace,
//                    c.execution_trace.n_rows * c.execution_trace.m_cols);

//     cudaMallocManaged(&p.random_coeff_powers, sizeof(qm31_t) * total_constraints);
//     cudaMallocManaged(&p.composition_poly, sizeof(qm31_t) * c.execution_trace.n_rows);
//     const auto start_0 = std::chrono::high_resolution_clock::now();
//     cudaMemPrefetchAsync(c.execution_trace.trace, c.execution_trace.n_rows * c.execution_trace.m_cols * sizeof(m31_t),
//                          0);
//     const auto stop_0 = std::chrono::high_resolution_clock::now();
//     const auto start_1 = std::chrono::high_resolution_clock::now();
//     compute_composition_polynomial(p, test_random_coeff);
//     const auto stop_1 = std::chrono::high_resolution_clock::now();
//     const auto start_2 = std::chrono::high_resolution_clock::now();
//     cudaMemPrefetchAsync(p.composition_poly, c.execution_trace.n_rows * sizeof(qm31_t), cudaCpuDeviceId);
//     const auto stop_2 = std::chrono::high_resolution_clock::now();

//     const auto duration_0 = std::chrono::duration_cast<std::chrono::milliseconds>(stop_0 - start_0);
//     const auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop_1 - start_1);
//     const auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop_2 - start_2);
//     std::cout << "h2d: " << duration_0.count() << " ms" << std::endl;
//     std::cout << "exec: " << duration_1.count() << " ms" << std::endl;
//     std::cout << "d2h: " << duration_2.count() << " ms" << std::endl;
//     return 0;
// }
