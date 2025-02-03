#include "fields/field_config.cuh"
using namespace field_config;

#include "fri.cu"
#include "utils/utils.h"
#include "fields/point.cuh"
#include "vec_ops/vec_ops.cuh"

#include "stwo.cuh"
// #include "api/m31.h"
#include "fields/stark_fields/m31.cuh"
#include "fib_eval_t.cuh"

namespace fri {
  /**
   * Extern "C" version of [fold_line](@ref fold_line) function with the following values of
   * template parameters (where the field is given by `-DFIELD` env variable during build):
   *  - `E` is the extension field type used for evaluations and alpha
   *  - `S` is the scalar field type used for domain elements
   * @param line_eval Pointer to the array of evaluations on the line
   * @param domain_elements Pointer to the array of domain elements
   * @param alpha The folding factor
   * @param folded_evals Pointer to the array where folded evaluations will be stored
   * @param n The number of evaluations
   * @param ctx The device context; if the stream is not 0, then everything is run async
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, fold_line)(
    scalar_t* line_eval1,
    scalar_t* line_eval2,
    scalar_t* line_eval3,
    scalar_t* line_eval4,
    scalar_t* domain_elements,
    q_extension_t alpha,
    scalar_t* folded_evals1,
    scalar_t* folded_evals2,
    scalar_t* folded_evals3,
    scalar_t* folded_evals4,
    uint64_t n,
    FriConfig& cfg)
  {
    return fri::fold_line(
      line_eval1, line_eval2, line_eval3, line_eval4, domain_elements, alpha, folded_evals1, folded_evals2,
      folded_evals3, folded_evals4, n, cfg);
  };

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, fold_line_new)(
    scalar_t* line_eval1,
    scalar_t* line_eval2,
    scalar_t* line_eval3,
    scalar_t* line_eval4,
    uint64_t line_domain_initial_index,
    uint32_t line_domain_log_size,
    q_extension_t alpha,
    scalar_t* folded_evals1,
    scalar_t* folded_evals2,
    scalar_t* folded_evals3,
    scalar_t* folded_evals4,
    uint64_t n,
    FriConfig& cfg)
  {
    line_t line_domain(line_domain_initial_index, line_domain_log_size);
    line_t test_domain(coset_t::half_odds(line_domain_log_size));
    scalar_t* domain_elements;
    line_domain.get_twiddles(&domain_elements);
    cfg.are_domain_elements_on_device = true;
    return fri::fold_line(
      line_eval1, line_eval2, line_eval3, line_eval4, domain_elements, alpha, folded_evals1, folded_evals2,
      folded_evals3, folded_evals4, n, cfg);
  };

  /**
   * Extern "C" version of [fold_circle_into_line](@ref fold_circle_into_line) function with the following values of
   * template parameters (where the field is given by `-DFIELD` env variable during build):
   *  - `E` is the extension field type used for evaluations and alpha
   *  - `S` is the scalar field type used for domain elements
   * @param circle_evals Pointer to the array of evaluations on the circle
   * @param domain_elements Pointer to the array of domain elements
   * @param alpha The folding factor
   * @param folded_line_evals Pointer to the array where folded evaluations will be stored
   * @param n The number of evaluations
   * @param ctx The device context; if the stream is not 0, then everything is run async
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, fold_circle_into_line)(
    scalar_t* circle_evals1,
    scalar_t* circle_evals2,
    scalar_t* circle_evals3,
    scalar_t* circle_evals4,
    scalar_t* domain_elements,
    q_extension_t alpha,
    scalar_t* folded_line_evals1,
    scalar_t* folded_line_evals2,
    scalar_t* folded_line_evals3,
    scalar_t* folded_line_evals4,
    uint64_t n,
    FriConfig& cfg)
  {
    return fri::fold_circle_into_line(
      circle_evals1, circle_evals2, circle_evals3, circle_evals4, domain_elements, alpha, folded_line_evals1,
      folded_line_evals2, folded_line_evals3, folded_line_evals4, n, cfg);
  };

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, fold_circle_into_line_new)(
    scalar_t* circle_evals1,
    scalar_t* circle_evals2,
    scalar_t* circle_evals3,
    scalar_t* circle_evals4,
    uint64_t domain_initial_index,
    uint32_t domain_log_size,
    q_extension_t alpha,
    scalar_t* folded_line_evals1,
    scalar_t* folded_line_evals2,
    scalar_t* folded_line_evals3,
    scalar_t* folded_line_evals4,
    uint64_t n,
    FriConfig& cfg)
  {
    domain_t test_domain(domain_log_size + 1);
    domain_t domain(coset_t(domain_initial_index, domain_log_size));
    scalar_t* domain_elements;
    domain.get_twiddles(&domain_elements);
    cfg.are_domain_elements_on_device = true;
    return fri::fold_circle_into_line(
      circle_evals1, circle_evals2, circle_evals3, circle_evals4, domain_elements, alpha, folded_line_evals1,
      folded_line_evals2, folded_line_evals3, folded_line_evals4, n, cfg);
  };

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, precompute_fri_twiddles)(uint32_t log_size)
  {
    CHK_INIT_IF_RETURN();
    for (uint32_t i = 2; i <= log_size; ++i) {
      coset_t coset = coset_t::half_odds(i);
      domain_t domain(coset);
      domain.compute_twiddles();
      line_t line_domain(coset);
      line_domain.compute_twiddles();
    }
    return CHK_LAST();
  };

  ///
// #include <cuda_runtime.h>
// #include <cooperative_groups.h>

  // Define a simple CUDA hash table (basic implementation)
  struct HashMapEntry {
    uint64_t key;
    scalar_t* value;
    int valid;
  };

  struct HashTable {
    HashMapEntry* table;
    int capacity;

    __device__ bool insert(uint64_t key, scalar_t* value) {
        int index = key % capacity;
        int start_index = index;  // To detect full table case

        do {
            if (table[index].valid == 0) { // Check if slot is free
                int old_valid = atomicExch(&table[index].valid, 1); // Mark as taken

                if (old_valid == 0) { // If successfully claimed, insert key-value
                    table[index].key = key;
                    table[index].value = value;
                    return true;
                }
            }
            index = (index + 1) % capacity; // Linear probing
        } while (index != start_index); // Stop if we wrap around

        return false; // Table is full
    }


    __device__ scalar_t* retrieve(uint64_t key)
    {
      int index = key % capacity;
      int start_index = index;

      do {
        if (!table[index].valid) return nullptr; // Stop if we reach an empty slot (not found)
        if (table[index].key == key) return table[index].value;

        index = (index + 1) % capacity;
      } while (index != start_index); // Stop if we wrap around

      return nullptr; // Not found
    }
  };

  // Persistent device hash table pointer
  __device__ HashTable* d_hash_table = nullptr;

  // Kernel to insert elements into the hash table
  __global__ void insertKernel(uint64_t key, scalar_t* value) { d_hash_table->insert(key, value); }

  // Kernel to retrieve elements from the hash table
  __global__ void retrieveKernel(uint64_t key, scalar_t* out_value)
  {
    out_value = d_hash_table->retrieve(key);
  }

  // CUDA function to initialize the hash table (only runs once)
  cudaError_t initialize_hash_table_once(int capacity)
  {
    static bool initialized = false;
    static cudaError_t lastStatus = cudaSuccess;

    if (initialized) return lastStatus; // Prevent reinitialization

    HashTable* h_hash_table;
    CHK_IF_RETURN(cudaMallocManaged(&h_hash_table, sizeof(HashTable)));
    CHK_IF_RETURN(cudaMallocManaged(&h_hash_table->table, capacity * sizeof(HashMapEntry)));

    for (int i = 0; i < capacity; i++) {
      h_hash_table->table[i].valid = false;
    }

    h_hash_table->capacity = capacity;
    CHK_IF_RETURN(cudaMemcpyToSymbol(d_hash_table, &h_hash_table, sizeof(HashTable*)));

    initialized = true;
    lastStatus = cudaSuccess;
    return cudaSuccess;
  }

  // CUDA function to retrieve elements from the hash table
  cudaError_t retrieve_trace(uint64_t size, scalar_t* d_out_data)
  {
    retrieveKernel<<<1, 1>>>(size, d_out_data);
    return CHK_LAST();
  }

  ///

  // CUDA function to insert elements into the hash table
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, preload_trace)(scalar_t* d_sub_trace_elements, uint64_t size)
  {
    CHK_INIT_IF_RETURN();

    CHK_IF_RETURN(initialize_hash_table_once(64)); // Initialize only once

    insertKernel<<<1, 1>>>(size, d_sub_trace_elements);

    return CHK_LAST();
  }

// #include <chrono>
// #include <fstream>
// #include <iostream>

  // #include "gpu-utils/device_context.cuh"

  static constexpr uint32_t BLOCK_SIZE = 256;

  // trace[3]
  // preprocessing
  // execution <---
  // interaction

  // m31_t execution_trace[row][col];

  __global__ void k_eval_row(
    const uint32_t total_constraints,
    const uint32_t m_cols,
    qm31_t* res,
    const qm31_t* random_coeff_powers,
    const m31_t* denom_inv,
    const m31_t* trace)
  {
    const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const m31_t* row = trace + thread_id * m_cols;
    fib_eval_t eval(random_coeff_powers, row, total_constraints, m_cols);
    eval.evaluate();
    const qm31_t row_res = eval.get_row_res();
    res[thread_id] = row_res * denom_inv[thread_id >> 20];
  }

  void generate_random_coeff_powers(qm31_t* random_coeff_powers, const qm31_t& x, const uint32_t n)
  {
    qm31_t acc = qm31_t::one();
    for (uint32_t i = 0; i < n; i++) {
      random_coeff_powers[i] = acc;
      acc = acc * x;
    }
  }

  m31_t coset_vanishing(const coset_t& coset, point_t p)
  {
    p = p - coset.initial_point + coset.at(coset.step_size >> 1);
    m31_t x = p.x;
    for (uint32_t i = 1; i < coset.log_size; i++) {
      x = point_t::dbl_x(x);
    }
    return x;
  }

  // static constexpr uint32_t BLOCK_SIZE = 256;

  void evaluate_constraint_quotients_on_domain(
    const prover_ctx_t& prover, const component_ctx_t& component
    // uint32_t eval_log_size,
    // uint32_t domain_log_size
  )
  {
    const uint32_t expand_log_size = component.eval_log_size - component.domain_log_size;
    const uint32_t domain_size = 1 << component.domain_log_size;
    const uint32_t eval_size = 1 << component.eval_log_size;
    const uint32_t expand_size = 1 << expand_log_size;
    const domain_t trace_domain(component.domain_log_size);
    const domain_t eval_domain(component.eval_log_size);
    m31_t* denom_inv;
    cudaMallocManaged(&denom_inv, sizeof(m31_t) * expand_size);

    // TODO: bit reverse
    for (int i = 0; i < expand_size; i++) {
      denom_inv[i] = m31_t::inverse(coset_vanishing(trace_domain.coset, eval_domain.at(i)));
    }

    // auto ctx = device_context::get_default_device_context();
    // BitReverseConfig& config = {
    //   ctx,   // ctx
    //   true,  // is_input_on_device
    //   true,  // is_output_on_device
    //   false, // is_async
    // };

    // bit_reverse_cuda(denom_inv, expand_size, config, denom_inv);

    // CPU version
    // for (int i = 0; i < component.execution_trace.n_rows; i++) {
    //     const m31_t *row = component.execution_trace.get_row(i);
    //     fib_eval_t eval(random_coeff_powers, row, total_constraints, component.execution_trace.m_cols);
    //     eval.evaluate();
    //     const qm31_t row_res = eval.get_row_res();
    //     res[i] = row_res * denom_inv[i >> component.domain_log_size];
    // }

    // GPU version
    k_eval_row<<<component.execution_trace.n_rows / BLOCK_SIZE, BLOCK_SIZE>>>(
      prover.total_constraints, component.execution_trace.m_cols, prover.composition_poly, prover.random_coeff_powers,
      denom_inv, component.execution_trace.trace);
    cudaDeviceSynchronize();
  }

  void compute_composition_polynomial(
    uint32_t total_constraints,
    uint32_t eval_log_size,
    uint32_t domain_log_size,
    uint32_t trace_rows_dimension,
    const qm31_t& random_coeff)
  {
    prover_ctx_t prover;
    prover.total_constraints = total_constraints;

    component_ctx_t component;
    component.eval_log_size = eval_log_size;
    component.domain_log_size = domain_log_size;

    // prover.components.push_back(component);

    component.interaction_trace.n_rows = trace_rows_dimension;
    component.interaction_trace.m_cols = 1 << eval_log_size;

    generate_random_coeff_powers(prover.random_coeff_powers, random_coeff, prover.total_constraints);

    retrieve_trace(eval_log_size, component.interaction_trace.trace);

    // for (uint32_t i = 0; i < prover.components.size(); i++) {
    evaluate_constraint_quotients_on_domain(prover, component);
    //}

    // for (uint32_t i = 0; i < total_constraints; i++) {
    //     qm31_t p = prover_ctx.secure_powers[i];
    //     printf("%x, %x, %x, %x\n", p.real.get_limb(), p.im1.get_limb(), p.im2.get_limb(), p.im3.get_limb());
    // }
  }

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

  // Example of how to use the kernel
  //  int main() {
  //      constexpr uint32_t total_constraints = 98;
  //      const qm31_t test_random_coeff = {0x583B16E6, 0x496EFFDF, 0xEBD5346, 0x40D4077F};

  //     prover_ctx_t p;
  //     p.total_constraints = total_constraints;
  //     component_ctx_t c;
  //     c.domain_log_size = 20;
  //     c.eval_log_size = 21;
  //     c.allocate_execution_trace(100, 2097152);
  //     p.components.push_back(c);

  //     // load test data
  //     readBinaryFile("/home/tonyw/scratch/stwo-playground/eval.bin", c.execution_trace.trace,
  //                    c.execution_trace.n_rows * c.execution_trace.m_cols);

  //     cudaMallocManaged(&p.random_coeff_powers, sizeof(qm31_t) * total_constraints);
  //     cudaMallocManaged(&p.composition_poly, sizeof(qm31_t) * c.execution_trace.n_rows);
  //     const auto start_0 = std::chrono::high_resolution_clock::now();
  //     cudaMemPrefetchAsync(c.execution_trace.trace, c.execution_trace.n_rows * c.execution_trace.m_cols *
  //     sizeof(m31_t),
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

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, compute_composition_polynomial)(
    // const prover_ctx_t& prover, const component_ctx_t& component
  )
  {
    CHK_INIT_IF_RETURN();
    //compute_composition_polynomial();

    return CHK_LAST();
  }

} // namespace fri
