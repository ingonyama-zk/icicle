#pragma once

// TODO Yuval: use current device context and stream
#include "cuda_runtime.h"
#include "appUtils/ntt/ntt.cuh"
#include "polynomials/gpu_backend/kernels.cuh"

namespace polynomials {

  /*============================== Polynomial GPU-context ==============================*/
  template <typename C, typename D, typename I>
  class GPUPolynomialContext : public IPolynomialContext<C, D, I>
  {
    typedef IPolynomialContext<C, D, I> PolyContext;
    using typename IPolynomialContext<C, D, I>::State;
    using IPolynomialContext<C, D, I>::ElementSize;

    // TODO Yuval:
    // (1) add device context (with stream, gpu id etc.)
    // (2) use streams and async ops
  public:
    ~GPUPolynomialContext() { release(); }

    static uint64_t ceil_to_power_of_two(uint64_t x) { return 1ULL << uint64_t(ceil(log2(x))); }

    void allocate(uint64_t nof_elements, State init_state) override
    {
      release();

      const uint64_t nof_elements_nearset_power_of_two = ceil_to_power_of_two(nof_elements);
      const uint64_t mem_size = nof_elements_nearset_power_of_two * ElementSize;
      cudaMalloc(&m_storage, mem_size);
      cudaMemset(m_storage, 0, mem_size);
      this->m_nof_elements = nof_elements_nearset_power_of_two;
      set_state(init_state);
    }

    void extend_mem_and_pad(uint64_t nof_elements)
    {
      const uint64_t nof_elements_nearset_power_of_two = ceil_to_power_of_two(nof_elements);
      const uint64_t new_mem_size = nof_elements_nearset_power_of_two * ElementSize;
      const uint64_t old_mem_size = this->m_nof_elements * ElementSize;

      void* new_storage;
      cudaMalloc(&new_storage, new_mem_size);
      cudaMemset(new_storage, 0, new_mem_size);

      cudaMemcpy(new_storage, m_storage, old_mem_size, cudaMemcpyDeviceToDevice);
      this->m_nof_elements = nof_elements_nearset_power_of_two;
      cudaFree(this->m_storage);
      this->m_storage = new_storage;
    }

    void release() override
    {
      if (m_storage != nullptr) { cudaFree(m_storage); }

      m_storage = nullptr;
      this->m_nof_elements = 0;
    }

    void set_state(State state) override { this->m_state = state; }

    C* init_from_coefficients(uint64_t nof_coefficients, const C* host_coefficients) override
    {
      allocate(nof_coefficients, State::Coefficients);
      if (host_coefficients) {
        cudaMemcpy(m_storage, host_coefficients, nof_coefficients * sizeof(C), cudaMemcpyHostToDevice);
      }
      return static_cast<C*>(m_storage);
    }

    I* init_from_rou_evaluations(uint64_t nof_evalutions, const I* host_evaluations) override
    {
      allocate(nof_evalutions, State::EvaluationsOnRou_Natural);
      if (host_evaluations) {
        cudaMemcpy(m_storage, host_evaluations, nof_evalutions * sizeof(C), cudaMemcpyHostToDevice);
      }
      return static_cast<I*>(m_storage);
    }

    std::pair<C*, uint64_t> get_coefficients() override
    {
      transform_to_coefficients();
      return std::make_pair(static_cast<C*>(m_storage), this->m_nof_elements);
    }

    std::pair<I*, uint64_t> get_rou_evaluations() override
    {
      transform_to_evaluations(0, false);
      return std::make_pair(static_cast<I*>(m_storage), this->m_nof_elements);
    }

    void transform_to_coefficients() override
    {
      const bool is_already_in_state = this->m_state == State::Coefficients;
      if (is_already_in_state) { return; }

      // transform from evaluations to coefficients
      auto ntt_config = ntt::DefaultNTTConfig<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;

      ntt_config.ordering =
        (this->m_state == State::EvaluationsOnRou_Natural) ? ntt::Ordering::kNN : ntt::Ordering::kRN;
      C* coeffs = static_cast<C*>(m_storage);
      I* evals = static_cast<I*>(m_storage);
      ntt::NTT(evals, this->m_nof_elements, ntt::NTTDir::kInverse, ntt_config, coeffs);
      set_state(State::Coefficients);
    }

    void transform_to_evaluations(uint64_t nof_evaluations, bool is_reversed) override
    {
      // TODO Yuval: can maybe optimize this
      nof_evaluations = (nof_evaluations == 0) ? this->m_nof_elements : ceil_to_power_of_two(nof_evaluations);
      const bool is_same_nof_evaluations = nof_evaluations == this->m_nof_elements;
      const bool is_same_order = is_reversed && this->m_state == State::EvaluationsOnRou_Reversed ||
                                 (!is_reversed && State::EvaluationsOnRou_Natural);
      const bool is_already_in_state = is_same_nof_evaluations && is_same_order;
      if (is_already_in_state) { return; }

      // There are 3 cases:
      // (1) coefficients to evaluations
      //    (1a) same size -> NTT (NR or NN)
      //    (1b) different_size -> alloc new mem, copy coeffs and NTT (NR or NN) inplace
      // (2) evaluations to evaluations (interpolation)
      //     alloc memory, INTT to coeffs then NTT back to evals (NR or NN)

      const bool is_allocate_new_mem = nof_evaluations > this->m_nof_elements;
      // allocate more memory and copy+pad
      if (is_allocate_new_mem) { extend_mem_and_pad(nof_evaluations); }

      auto ntt_config = ntt::DefaultNTTConfig<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;

      C* coeffs = static_cast<C*>(m_storage);
      I* evals = static_cast<I*>(m_storage);
      const bool is_coeffs_to_evals = this->m_state == State::Coefficients;
      if (is_coeffs_to_evals) {
        // already copied the coefficients with padding. Now computing evaluations.
        ntt_config.ordering = is_reversed ? ntt::Ordering::kNR : ntt::Ordering::kNN;
        ntt::NTT(coeffs, nof_evaluations, ntt::NTTDir::kForward, ntt_config, evals);
      } else {
        // interpolation: transform to coefficients and back
        transform_to_coefficients();
        ntt_config.ordering = is_reversed ? ntt::Ordering::kNR : ntt::Ordering::kNN;
        ntt::NTT(coeffs, nof_evaluations, ntt::NTTDir::kForward, ntt_config, evals);
      }

      this->m_nof_elements = nof_evaluations;
      set_state(is_reversed ? State::EvaluationsOnRou_Natural : State::EvaluationsOnRou_Natural);
    }

    void print(std::ostream& os) override
    {
      if (this->get_state() == State::Coefficients) {
        print_coeffs(os);
      } else {
        print_evals(os);
      }
    }

    void print_coeffs(std::ostream& os) const
    {
      auto host_coeffs = std::make_unique<C[]>(this->m_nof_elements);
      cudaMemcpy(host_coeffs.get(), m_storage, this->m_nof_elements * sizeof(C), cudaMemcpyDeviceToHost);

      os << "(id=" << PolyContext::m_id << ")[";
      for (size_t i = 0; i < this->m_nof_elements; ++i) {
        os << host_coeffs.get()[i];
        if (i < this->m_nof_elements - 1) { os << ", "; }
      }
      os << "] (state=coefficients)" << std::endl;
    }

    void print_evals(std::ostream& os) const
    {
      auto host_evals = std::make_unique<I[]>(this->m_nof_elements);
      cudaMemcpy(host_evals.get(), m_storage, this->m_nof_elements * sizeof(I), cudaMemcpyDeviceToHost);

      os << "(id=" << PolyContext::m_id << ")[";
      for (size_t i = 0; i < this->m_nof_elements; ++i) {
        os << host_evals.get()[i];
        if (i < this->m_nof_elements - 1) { os << ", "; }
      }

      if (this->get_state() == State::EvaluationsOnRou_Reversed) {
        os << "] (state=rou evaluations Reversed)";
      } else {
        os << "] (state=rou evaluations )";
      }
    }

  private:
    // Members
    void* m_storage = nullptr;
  };

  /*============================== Polynomial GPU-backend ==============================*/

  template <typename C, typename D, typename I, typename ECpoint>
  class GPUPolynomialBackend : public IPolynomialBackend<C, D, I, ECpoint>
  {
    typedef IPolynomialContext<C, D, I> PolyContext;
    typedef typename IPolynomialContext<C, D, I>::State State;

  public:
    void add_sub(PolyContext& res, PolyContext& a, PolyContext& b, bool add1_sub0)
    {
      // TODO Yuval: can do it evaluations too but need to make sure same #evaluations (on ROU)
      auto [a_coeff_p, a_nof_coeff] = a.get_coefficients();
      auto [b_coeff_p, b_nof_coeff] = b.get_coefficients();

      const auto res_nof_coeff = max(a_nof_coeff, b_nof_coeff);
      res.allocate(res_nof_coeff, State::Coefficients);
      auto [res_coeff_p, _] = res.get_coefficients();

      const int NOF_THREADS = 32;
      const int NOF_BLOCKS = (res_nof_coeff + NOF_THREADS - 1) / NOF_THREADS;
      AddSubKernel<<<NOF_BLOCKS, NOF_THREADS>>>(a_coeff_p, b_coeff_p, a_nof_coeff, b_nof_coeff, add1_sub0, res_coeff_p);
    }

    void add(PolyContext& res, PolyContext& a, PolyContext& b) override { add_sub(res, a, b, true /*=add*/); }
    void subtract(PolyContext& res, PolyContext& a, PolyContext& b) override { add_sub(res, a, b, false /*=sub*/); }

    void multiply(PolyContext& c, PolyContext& a, PolyContext& b) override
    {
      const uint64_t nof_

      // const uint32_t a_N = a.get_nof_elements();
      // const uint32_t b_N = b.get_nof_elements();
      // const uint32_t N = max(a_N, b_N);

      // // (1) transform a,b to coefficients such that both have N coefficients
      // auto [a_coeff_p, a_nof_coeff] = a.get_coefficients(N);
      // auto [b_coeff_p, b_nof_coeff] = b.get_coefficients(N);

      // // (2) allocate c (c=a*b)
      // const auto c_N = 2 * N;
      // I* c_evals_low_p = c.init_from_rou_evaluations(c_N);
      // I* c_evals_high_p = c_evals_low_p + N;

      // std::cout << "a_N=" << a_N << ", b_N=" << b_N << ", c_N=" << c_N << "\n";

      // // (3) compute NTT of a,b on coset and write to c
      // auto ntt_config = ntt::DefaultNTTConfig<C>();
      // ntt_config.are_inputs_on_device = true;
      // ntt_config.are_outputs_on_device = true;
      // ntt_config.ordering = ntt::Ordering::kNR;
      // // ntt_config.coset_gen =
      // // test_type::omega(c_N); // TODO Yuval: MUST USE THE ROOT CORRESPONDING TO THE ONE IN INIT DOMAIN!!!

      // c.print(std::cout);
      // ntt::NTT(a_coeff_p, N, ntt::NTTDir::kForward, ntt_config, c_evals_low_p);  // a_H1
      // ntt::NTT(a_coeff_p, N, ntt::NTTDir::kForward, ntt_config, c_evals_high_p); // b_H1

      // a.print(std::cout);
      // b.print(std::cout);
      // c.print(std::cout);

      // // (4) compute a_H1 * b_H1 inplace
      // const int NOF_THREADS = 32;
      // const int NOF_BLOCKS = (N + NOF_THREADS - 1) / NOF_THREADS;
      // Mul<<<NOF_BLOCKS, NOF_THREADS>>>(c_evals_low_p, c_evals_high_p, N, c_evals_high_p);

      // // (5) transform a,b to evaluations
      // auto [a_evals_p, a_nof_evals] = a.get_rou_evaluations(N);
      // auto [b_evals_p, b_nof_evals] = b.get_rou_evaluations(N);
      // a.print(std::cout);
      // b.print(std::cout);
      // c.print(std::cout);

      // // (6) compute a_H0 * b_H0
      // Mul<<<NOF_BLOCKS, NOF_THREADS>>>(a_evals_p, b_evals_p, N, c_evals_low_p);
      // std::cout << "finally\n";
      // c.print(std::cout);
      // std::cout << "MUL DONE\n";
      // c.print(std::cout);
    }
    void divide(PolyContext& Quotient_out, PolyContext& Remainder_out, PolyContext& op_a, PolyContext& op_b) override
    {
      throw std::runtime_error("not implemented yet");
    }
    void quotient(PolyContext& out, PolyContext& op_a, PolyContext& op_b) override
    {
      throw std::runtime_error("not implemented yet");
    }
    void remainder(PolyContext& out, PolyContext& op_a, PolyContext& op_b) override
    {
      throw std::runtime_error("not implemented yet");
    }
    void divide_by_vanishing_polynomial(PolyContext& out, PolyContext& op_a, uint32_t vanishing_poly_degree) override
    {
      throw std::runtime_error("not implemented yet");
    }

    // arithmetic with monomials
    void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint32_t monomial) override
    {
      throw std::runtime_error("not implemented yet");
    }
    void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint32_t monomial) override
    {
      throw std::runtime_error("not implemented yet");
    }

    // dot product with coefficients
    ECpoint dot_product_with_coefficients(PolyContext& op, ECpoint* points, uint32_t nof_points) override
    {
      return ECpoint::zero();
    }

    int32_t degree(PolyContext& p) override
    {
      auto [coeff, nof_coeff] = p.get_coefficients();

      int32_t* d_degree;
      int32_t h_degree;
      cudaMalloc(&d_degree, sizeof(int32_t));
      // TODO Yuval parallelize kernel
      HighestNonZeroIdx<<<1, 1>>>(coeff, nof_coeff, d_degree);
      cudaMemcpy(&h_degree, d_degree, sizeof(int32_t), cudaMemcpyDeviceToHost);
      cudaFree(d_degree);

      return h_degree + 1;
    }

    I evaluate(PolyContext& p, const D& domain_x) override
    {
      auto [coeff, nof_coeff] = p.get_coefficients();
      I *d_evaluation, *d_domain_x;
      I* d_tmp;
      cudaMalloc(&d_evaluation, sizeof(I));
      cudaMalloc(&d_domain_x, sizeof(I));
      cudaMemcpy(d_domain_x, &domain_x, sizeof(I), cudaMemcpyHostToDevice);
      cudaMalloc(&d_tmp, sizeof(I) * nof_coeff);
      const int NOF_THREADS = 32;
      const int NOF_BLOCKS = (nof_coeff + NOF_THREADS - 1) / NOF_THREADS;
      // TODO Yuval: parallelize kernel
      evalutePolynomialWithoutReduction<<<NOF_BLOCKS, NOF_THREADS>>>(domain_x, coeff, nof_coeff, d_tmp);
      dummyReduce<<<1, 1>>>(d_tmp, nof_coeff, d_evaluation);

      I h_evaluation;
      cudaMemcpy(&h_evaluation, d_evaluation, sizeof(I), cudaMemcpyDeviceToHost);
      cudaFree(d_evaluation);
      cudaFree(d_domain_x);
      cudaFree(d_tmp);

      return h_evaluation;
    }
    void evaluate(PolyContext& p, const D* domain_x, uint32_t nof_domain_points, I* evaluations /*OUT*/) override
    {
      throw std::runtime_error("not implemented yet");
    }

    C get_coefficient(PolyContext& op, uint32_t coeff_idx) override
    {
      throw std::runtime_error("not implemented yet");
      return C::zero();
    }
    uint32_t get_coefficients(PolyContext& op, C* coefficients) override
    {
      throw std::runtime_error("not implemented yet");
      return 0;
    }
  };
} // namespace polynomials