#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"
#include "icicle/sumcheck/sumcheck_transcript.h"
#include "cpu_program_executor.h"
#include "icicle/backend/sumcheck_backend.h"
#include "taskflow/taskflow.hpp"

namespace icicle {
  template <typename F>
  class CpuSumcheckBackend : public SumcheckBackend<F>
  {
  public:
    CpuSumcheckBackend() : SumcheckBackend<F>() {}

    // Calculates a proof for the mle polynomials
    eIcicleError get_proof(
      const std::vector<F*>& mle_polynomials,
      const uint64_t mle_polynomial_size,
      const F& claimed_sum,
      const CombineFunction<F>& combine_function,
      const SumcheckTranscriptConfig<F>&& transcript_config,
      const SumcheckConfig& sumcheck_config,
      SumcheckProof<F>& sumcheck_proof /*out*/) override
    {
      if (sumcheck_config.use_extension_field) {
        ICICLE_LOG_ERROR << "SumcheckConfig::use_extension_field field = true is currently unsupported";
        return eIcicleError::INVALID_ARGUMENT;
      }
      // Allocate memory for the intermediate calculation: the folded mle polynomials
      const int nof_mle_poly = mle_polynomials.size();
      std::vector<F*> folded_mle_polynomials(nof_mle_poly); // folded mle_polynomials with the same format as inputs
      std::vector<F> folded_mle_polynomials_values(
        nof_mle_poly * mle_polynomial_size); // folded_mle_polynomials data itself
      // init the folded_mle_polynomials pointers
      for (int poly_idx = 0; poly_idx < nof_mle_poly; poly_idx++) {
        folded_mle_polynomials[poly_idx] = &(folded_mle_polynomials_values[poly_idx * mle_polynomial_size]);
      }

      // Check that the combine function has a legal polynomial degree
      const int combine_function_poly_degree = combine_function.get_polynomial_degree();
      if (combine_function_poly_degree < 0) {
        ICICLE_LOG_ERROR << "Illegal polynomial degree (" << combine_function_poly_degree
                         << ") for provided combine function";
        return eIcicleError::INVALID_ARGUMENT;
      }

      // create sumcheck_transcript for the Fiat-Shamir
      const uint32_t combine_function_poly_degree_u = combine_function_poly_degree;
      const uint32_t nof_rounds = std::log2(mle_polynomial_size);
      SumcheckTranscript<F> sumcheck_transcript(
        claimed_sum, nof_rounds, combine_function_poly_degree_u, std::move(transcript_config));

      // reset the sumcheck proof to accumulate the round polynomials
      sumcheck_proof.init(nof_rounds, combine_function_poly_degree_u);

      // set threads values
      const int max_nof_workers = get_nof_workers(sumcheck_config);
      const int round_polynomial_size = sumcheck_proof.get_round_polynomial(0).size();
      std::vector<std::vector<F>> worker_round_polynomial(
        max_nof_workers, std::vector<F>(round_polynomial_size, F::zero()));

      // run log2(poly_size) rounds
      int cur_mle_polynomial_size = mle_polynomial_size;
      for (int round_idx = 0; round_idx < nof_rounds; ++round_idx) {
        const int nof_total_iterations = (round_idx == 0) ? cur_mle_polynomial_size / 2 : cur_mle_polynomial_size / 4;
        const int nof_workers = std::min(max_nof_workers, nof_total_iterations);
        const int nof_tasks_per_worker = (nof_total_iterations + nof_workers - 1) / nof_workers; // round up
        F alpha =
          round_idx ? sumcheck_transcript.get_alpha(sumcheck_proof.get_round_polynomial(round_idx - 1)) : F::zero();
        for (int worker_idx = 0; worker_idx < nof_workers; ++worker_idx) {
          m_taskflow.emplace([&, worker_idx]() {
            const int start_element_idx = worker_idx * nof_tasks_per_worker;
            const int cur_worker_nof_tasks = std::min(nof_tasks_per_worker, nof_total_iterations - start_element_idx);
            switch (round_idx) {
            case 0:
              build_round_polynomial_0(
                mle_polynomials, cur_mle_polynomial_size, start_element_idx, cur_worker_nof_tasks, combine_function,
                worker_round_polynomial[worker_idx]);
              break;
            case 1:
              fold_and_build_round_polynomial_1(
                alpha, mle_polynomials, folded_mle_polynomials, cur_mle_polynomial_size, start_element_idx,
                cur_worker_nof_tasks, combine_function, worker_round_polynomial[worker_idx]);
              break;
            default:
              fold_and_build_round_polynomial_i(
                alpha, folded_mle_polynomials, cur_mle_polynomial_size, start_element_idx, cur_worker_nof_tasks,
                combine_function, worker_round_polynomial[worker_idx]);
            }
          });
        }
        m_executor.run(m_taskflow).wait();
        m_taskflow.clear();

        // increment folded_mle_polynomials to point to the generated data
        if (round_idx > 1) {
          for (int mle_polynomial_idx = 0; mle_polynomial_idx < nof_mle_poly; mle_polynomial_idx++) {
            folded_mle_polynomials[mle_polynomial_idx] += cur_mle_polynomial_size;
          }
        }
        cur_mle_polynomial_size = round_idx ? cur_mle_polynomial_size / 2 : cur_mle_polynomial_size;

        std::vector<F>& round_polynomial = sumcheck_proof.get_round_polynomial(round_idx);
        for (int worker_idx = 0; worker_idx < nof_workers; ++worker_idx) {
          for (int k = 0; k < round_polynomial.size(); ++k) {
            round_polynomial[k] = round_polynomial[k] + worker_round_polynomial[worker_idx][k];
            worker_round_polynomial[worker_idx][k] = F::zero();
          }
        }
      }
      return eIcicleError::SUCCESS;
    }

  private:
    // members
    tf::Taskflow m_taskflow; // Accumulate tasks
    tf::Executor m_executor; // execute all tasks accumulated on multiple threads

    // functions
    int get_nof_workers(const SumcheckConfig& config) const
    {
      int nof_workers = config.ext && config.ext->has("n_threads") ? config.ext->get<int>("n_threads")
                                                                   : // number of threads provided by config
                          std::thread::hardware_concurrency();       // check machine properties
      if (nof_workers <= 0) {
        ICICLE_LOG_WARNING << "Unable to detect number of hardware supported threads - fixing it to 1\n";
        nof_workers = 1;
      }
      return nof_workers;
    }

    void build_round_polynomial_0(
      const std::vector<F*>& in_mle_polynomials,
      const uint64_t mle_polynomial_size,
      const int start_element_idx,
      const int nof_iterations,
      const CombineFunction<F>& combine_function,
      std::vector<F>& round_polynomial)
    {
      // init program_executor input pointers
      const int nof_polynomials = in_mle_polynomials.size();
      std::vector<F> combine_func_inputs(nof_polynomials);
      CpuProgramExecutor program_executor(combine_function);

      for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
        program_executor.m_variable_ptrs[poly_idx] = &(combine_func_inputs[poly_idx]);
      }
      // init m_program_executor output pointer
      F combine_func_result;
      program_executor.m_variable_ptrs[nof_polynomials] = &combine_func_result;

      for (int element_idx = start_element_idx; element_idx < start_element_idx + nof_iterations; ++element_idx) {
        for (int k = 0; k < round_polynomial.size(); ++k) {
          // update the combine program inputs for k
          for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
            combine_func_inputs[poly_idx] =                              // (1-k)*element_i + k*element_i+1
              (k == 0) ? in_mle_polynomials[poly_idx][2 * element_idx] : // for k=0: = element_i
                (k == 1) ? in_mle_polynomials[poly_idx][2 * element_idx + 1]
                         :                                           // for k=1: = element_i+1
                combine_func_inputs[poly_idx] -                      // else = prev result
                  in_mle_polynomials[poly_idx][2 * element_idx] +    //        - element_i
                  in_mle_polynomials[poly_idx][2 * element_idx + 1]; //        + element_i+1
          }
          // execute the combine functions and append to the round polynomial
          program_executor.execute();
          round_polynomial[k] = round_polynomial[k] + combine_func_result;
        }
      }
    }

    void fold_and_build_round_polynomial_1(
      const F& alpha,
      const std::vector<F*>& in_mle_polynomials,
      std::vector<F*>& folded_mle_polynomials,
      const uint64_t mle_polynomial_size,
      const int start_element_idx,
      const int nof_iterations,
      const CombineFunction<F>& combine_function,
      std::vector<F>& round_polynomial)
    {
      // init program_executor input pointers
      const int nof_polynomials = in_mle_polynomials.size();
      std::vector<F> combine_func_inputs(nof_polynomials);
      CpuProgramExecutor program_executor(combine_function);

      for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
        program_executor.m_variable_ptrs[poly_idx] = &(combine_func_inputs[poly_idx]);
      }
      // init m_program_executor output pointer
      F combine_func_result;
      program_executor.m_variable_ptrs[nof_polynomials] = &combine_func_result;

      for (int element_idx = start_element_idx; element_idx < start_element_idx + nof_iterations; ++element_idx) {
        // fold in_mle_polynomials vector to folded_mle_polynomials vector
        for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
          // fold element(2i) and element(2i+1) -> folded(i)
          folded_mle_polynomials[poly_idx][2 * element_idx] =
            in_mle_polynomials[poly_idx][4 * element_idx] +
            alpha * (in_mle_polynomials[poly_idx][4 * element_idx + 1] - in_mle_polynomials[poly_idx][4 * element_idx]);

          // fold element(i+n/4) and element(i+3n/4) -> folded(2*i+1)
          folded_mle_polynomials[poly_idx][2 * element_idx + 1] =
            in_mle_polynomials[poly_idx][4 * element_idx + 2] +
            alpha *
              (in_mle_polynomials[poly_idx][4 * element_idx + 3] - in_mle_polynomials[poly_idx][4 * element_idx + 2]);
        }

        for (int k = 0; k < round_polynomial.size(); ++k) {
          // update the combine program inputs for k
          for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
            combine_func_inputs[poly_idx] =                                  // (1-k)*element_i + k*element_i+1
              (k == 0) ? folded_mle_polynomials[poly_idx][2 * element_idx] : // for k=0: = element_i
                (k == 1) ? folded_mle_polynomials[poly_idx][2 * element_idx + 1]
                         :                                               // for k=1: = element_i+1
                combine_func_inputs[poly_idx] -                          // else = prev result
                  folded_mle_polynomials[poly_idx][2 * element_idx] +    //        - element_i
                  folded_mle_polynomials[poly_idx][2 * element_idx + 1]; //        + element_i+1
          }
          // execute the combine functions and append to the round polynomial
          program_executor.execute();
          round_polynomial[k] = round_polynomial[k] + combine_func_result;
        }
      }
    }

    void fold_and_build_round_polynomial_i(
      const F& alpha,
      std::vector<F*>& folded_mle_polynomials,
      const uint64_t mle_polynomial_size,
      const int start_element_idx,
      const int nof_iterations,
      const CombineFunction<F>& combine_function,
      std::vector<F>& round_polynomial)
    {
      // init program_executor input pointers
      const int nof_polynomials = folded_mle_polynomials.size();
      std::vector<F> combine_func_inputs(nof_polynomials);
      CpuProgramExecutor program_executor(combine_function);

      for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
        program_executor.m_variable_ptrs[poly_idx] = &(combine_func_inputs[poly_idx]);
      }
      // init m_program_executor output pointer
      F combine_func_result;
      program_executor.m_variable_ptrs[nof_polynomials] = &combine_func_result;

      for (int element_idx = start_element_idx; element_idx < start_element_idx + nof_iterations; ++element_idx) {
        // fold folded_mle_polynomials vector to itself
        for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
          // fold folded(2*i) and folded(2*i+1) -> folded(2*i)
          folded_mle_polynomials[poly_idx][mle_polynomial_size + 2 * element_idx] =
            folded_mle_polynomials[poly_idx][4 * element_idx] +
            alpha * (folded_mle_polynomials[poly_idx][4 * element_idx + 1] -
                     folded_mle_polynomials[poly_idx][4 * element_idx]);

          // fold folded(2*i+2) and folded(2*i+3) -> folded(2*i+1)
          folded_mle_polynomials[poly_idx][mle_polynomial_size + 2 * element_idx + 1] =
            folded_mle_polynomials[poly_idx][4 * element_idx + 2] +
            alpha * (folded_mle_polynomials[poly_idx][4 * element_idx + 3] -
                     folded_mle_polynomials[poly_idx][4 * element_idx + 2]);
        }

        // update_round_polynomial(element_idx, folded_mle_polynomials, mle_polynomial_size, program_executor,
        // round_polynomial);
        for (int k = 0; k < round_polynomial.size(); ++k) {
          // update the combine program inputs for k
          for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
            combine_func_inputs[poly_idx] = // (1-k)*element_i + k*element_i+1
              (k == 0) ? folded_mle_polynomials[poly_idx][mle_polynomial_size + 2 * element_idx]
                       : // for k=0: = element_i
                (k == 1) ? folded_mle_polynomials[poly_idx][mle_polynomial_size + 2 * element_idx + 1]
                         :                                                                     // for k=1: = element_i+1
                combine_func_inputs[poly_idx] -                                                // else = prev result
                  folded_mle_polynomials[poly_idx][mle_polynomial_size + 2 * element_idx] +    //        - element_i
                  folded_mle_polynomials[poly_idx][mle_polynomial_size + 2 * element_idx + 1]; //        + element_i+1
          }
          // execute the combine functions and append to the round polynomial
          program_executor.execute();
          round_polynomial[k] = round_polynomial[k] + combine_func_result;
        }
      }
    }
  };

} // namespace icicle
