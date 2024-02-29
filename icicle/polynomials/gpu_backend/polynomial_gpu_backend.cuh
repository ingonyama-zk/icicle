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

    void allocate(uint64_t nof_elements, State init_state, bool memset_zeros) override
    {
      release();

      const uint64_t nof_elements_nearset_power_of_two = ceil_to_power_of_two(nof_elements);
      const uint64_t mem_size = nof_elements_nearset_power_of_two * ElementSize;
      cudaMalloc(&m_storage, mem_size);
      this->m_nof_elements = nof_elements_nearset_power_of_two;
      set_state(init_state);
      if (memset_zeros) { cudaMemset(m_storage, 0, mem_size); }
    }

    void* allocate_mem(uint64_t nof_elements, bool is_set_zeros = true)
    {
      const uint64_t nof_elements_nearset_power_of_two = ceil_to_power_of_two(nof_elements);
      const uint64_t new_mem_size = nof_elements_nearset_power_of_two * ElementSize;

      void* new_storage;
      cudaMalloc(&new_storage, new_mem_size);

      if (is_set_zeros) { cudaMemset(new_storage, 0, new_mem_size); }
      return new_storage;
    }

    void set_storage(void* storage, uint64_t nof_elements)
    {
      release();
      m_storage = storage;
      this->m_nof_elements = nof_elements;
    }

    void extend_mem_and_pad(uint64_t nof_elements)
    {
      const uint64_t nof_elements_nearset_power_of_two = ceil_to_power_of_two(nof_elements);
      const uint64_t new_mem_size = nof_elements_nearset_power_of_two * ElementSize;
      const uint64_t old_mem_size = this->m_nof_elements * ElementSize;

      void* new_storage = allocate_mem(nof_elements);
      cudaMemcpy(new_storage, m_storage, old_mem_size, cudaMemcpyDeviceToDevice);
      set_storage(new_storage, nof_elements_nearset_power_of_two);
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
      const bool is_memset_zeros = host_coefficients == nullptr;
      allocate(nof_coefficients, State::Coefficients, is_memset_zeros);
      if (host_coefficients) {
        cudaMemcpy(m_storage, host_coefficients, nof_coefficients * sizeof(C), cudaMemcpyHostToDevice);
      }
      return static_cast<C*>(m_storage);
    }

    I* init_from_rou_evaluations(uint64_t nof_evalutions, const I* host_evaluations) override
    {
      const bool is_memset_zeros = host_evaluations == nullptr;
      allocate(nof_evalutions, State::EvaluationsOnRou_Natural, is_memset_zeros);
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
      const bool is_reversed = this->m_state == State::EvaluationsOnRou_Reversed;
      transform_to_evaluations(0, is_reversed);
      return std::make_pair(static_cast<I*>(m_storage), this->m_nof_elements);
    }

    void transform_to_coefficients(uint64_t nof_coefficients = 0) override
    {
      // cannot really get more coefficients but sometimes want to pad for NTT. In that case
      // nof_coefficients>m_nof_elements
      nof_coefficients = (nof_coefficients == 0) ? this->m_nof_elements : ceil_to_power_of_two(nof_coefficients);
      const bool is_same_nof_coefficients = this->m_nof_elements == nof_coefficients;
      const bool is_already_in_state = this->m_state == State::Coefficients && is_same_nof_coefficients;
      if (is_already_in_state) { return; }

      const bool is_already_in_coeffs = this->m_state == State::Coefficients;
      // case 1: already in coefficients. Need to allocate larger memory and zero pad
      if (is_already_in_coeffs) {
        extend_mem_and_pad(nof_coefficients);
        return;
      }

      // case 2: transform from evaluations. May need to allocate larger memory
      I* evals = static_cast<I*>(m_storage);
      C* coeffs = static_cast<C*>(m_storage);
      const bool is_allocate_new_mem = nof_coefficients > this->m_nof_elements;
      if (is_allocate_new_mem) {
        void* new_mem = allocate_mem(nof_coefficients);
        coeffs = static_cast<C*>(new_mem);
      }

      // transform from evaluations to coefficients
      auto ntt_config = ntt::DefaultNTTConfig<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;

      ntt_config.ordering =
        (this->m_state == State::EvaluationsOnRou_Natural) ? ntt::Ordering::kNN : ntt::Ordering::kRN;
      ntt::NTT(evals, this->m_nof_elements, ntt::NTTDir::kInverse, ntt_config, coeffs);
      set_state(State::Coefficients);

      if (is_allocate_new_mem) { set_storage(coeffs, nof_coefficients); } // release old memory and use new
    }

    void transform_to_evaluations(uint64_t nof_evaluations = 0, bool is_reversed = false) override
    {
      // TODO Yuval: can maybe optimize this
      nof_evaluations = (nof_evaluations == 0) ? this->m_nof_elements : ceil_to_power_of_two(nof_evaluations);
      const bool is_same_nof_evaluations = nof_evaluations == this->m_nof_elements;
      const bool is_same_order = is_reversed && this->m_state == State::EvaluationsOnRou_Reversed ||
                                 (!is_reversed && State::EvaluationsOnRou_Natural);
      const bool is_already_in_state = is_same_nof_evaluations && is_same_order;
      if (is_already_in_state) { return; }

      // TODO Yuval: evaluations->evaluations with different ordering can be implemented via inplace reorder more
      // efficiently than it is now

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
      set_state(is_reversed ? State::EvaluationsOnRou_Reversed : State::EvaluationsOnRou_Natural);
    }

    void print(std::ostream& os) override
    {
      if (this->get_state() == State::Coefficients) {
        print_coeffs(os);
      } else {
        print_evals(os);
      }
    }

    void print_coeffs(std::ostream& os)
    {
      transform_to_coefficients();
      auto host_coeffs = std::make_unique<C[]>(this->m_nof_elements);
      cudaMemcpy(host_coeffs.get(), m_storage, this->m_nof_elements * sizeof(C), cudaMemcpyDeviceToHost);

      os << "(id=" << PolyContext::m_id << ")[";
      for (size_t i = 0; i < this->m_nof_elements; ++i) {
        os << host_coeffs[i];
        if (i < this->m_nof_elements - 1) { os << ", "; }
      }
      os << "] (state=coefficients)" << std::endl;
    }

    void print_evals(std::ostream& os)
    {
      transform_to_evaluations();
      auto host_evals = std::make_unique<I[]>(this->m_nof_elements);
      cudaMemcpy(host_evals.get(), m_storage, this->m_nof_elements * sizeof(I), cudaMemcpyDeviceToHost);

      os << "(id=" << PolyContext::m_id << ")[";
      for (size_t i = 0; i < this->m_nof_elements; ++i) {
        os << host_evals[i];
        if (i < this->m_nof_elements - 1) { os << ", "; }
      }

      if (this->get_state() == State::EvaluationsOnRou_Reversed) {
        os << "] (state=rou evaluations Reversed)" << std::endl;
      } else {
        os << "] (state=rou evaluations )" << std::endl;
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
      // TODO Yuval: can do it evaluations too if same #evaluations (on ROU)
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
      const bool is_multiply_with_cosets = true;
      if (is_multiply_with_cosets) { return multiply_with_cosets(c, a, b); }
      return multiply_with_padding(c, a, b);
    }

    void multiply_with_padding(PolyContext& c, PolyContext& a, PolyContext& b)
    {
      const uint64_t a_N_orig = a.get_nof_elements();
      const uint64_t b_N_orig = b.get_nof_elements();
      const uint64_t N = max(a_N_orig, b_N_orig);
      const uint64_t c_N = 2 * N;

      // (1) transform a,b to 2N evaluations
      a.transform_to_evaluations(c_N, true /*=reversed*/);
      b.transform_to_evaluations(c_N, true /*=reversed*/);
      auto [a_evals_p, a_N] = a.get_rou_evaluations();
      auto [b_evals_p, b_N] = b.get_rou_evaluations();

      // (2) allocate c (c=a*b) and compute element-wise multiplication on evaluations
      c.allocate(c_N, State::EvaluationsOnRou_Reversed, false /*=memset zeros*/);
      auto [c_evals_p, _] = c.get_rou_evaluations();

      const int NOF_THREADS = 32;
      const int NOF_BLOCKS = (c_N + NOF_THREADS - 1) / NOF_THREADS;
      Mul<<<NOF_BLOCKS, NOF_THREADS>>>(a_evals_p, b_evals_p, c_N, c_evals_p);
    }

    void multiply_with_cosets(PolyContext& c, PolyContext& a, PolyContext& b)
    {
      const uint64_t a_N = a.get_nof_elements();
      const uint64_t b_N = b.get_nof_elements();
      const uint64_t N = max(a_N, b_N);

      // (1) transform a,b to coefficients such that both have N coefficients
      a.transform_to_coefficients(N);
      b.transform_to_coefficients(N);
      auto [a_coeff_p, _] = a.get_coefficients();
      auto [b_coeff_p, __] = b.get_coefficients();
      // (2) allocate c (c=a*b)
      const uint64_t c_N = 2 * N;
      c.allocate(c_N, State::EvaluationsOnRou_Reversed, false /*=memset zeros*/);
      auto [c_evals_low_p, ___] = c.get_rou_evaluations();
      I* c_evals_high_p = c_evals_low_p + N;

      // (3) compute NTT of a,b on coset and write to c
      auto ntt_config = ntt::DefaultNTTConfig<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.ordering = ntt::Ordering::kNR;
      ntt_config.coset_gen = ntt::GetRootOfUnity<C>((uint64_t)log2(c_N), ntt_config.ctx);

      ntt::NTT(a_coeff_p, N, ntt::NTTDir::kForward, ntt_config, c_evals_low_p);  // a_H1
      ntt::NTT(b_coeff_p, N, ntt::NTTDir::kForward, ntt_config, c_evals_high_p); // b_H1

      // (4) compute a_H1 * b_H1 inplace
      const int NOF_THREADS = 32;
      const int NOF_BLOCKS = (N + NOF_THREADS - 1) / NOF_THREADS;
      Mul<<<NOF_BLOCKS, NOF_THREADS>>>(c_evals_low_p, c_evals_high_p, N, c_evals_high_p);
      // (5) transform a,b to evaluations
      a.transform_to_evaluations(N, true /*=reversed*/);
      b.transform_to_evaluations(N, true /*=reversed*/);
      auto [a_evals_p, a_nof_evals] = a.get_rou_evaluations();
      auto [b_evals_p, b_nof_evals] = b.get_rou_evaluations();

      // (6) compute a_H0 * b_H0
      Mul<<<NOF_BLOCKS, NOF_THREADS>>>(a_evals_p, b_evals_p, N, c_evals_low_p);
    }

    void divide(PolyContext& Q /*OUT*/, PolyContext& R /*OUT*/, PolyContext& a, PolyContext& b) override
    {
      auto [a_coeffs, a_N] = a.get_coefficients();
      auto [b_coeffs, b_N] = b.get_coefficients();

      // TODO Yuval: assert deg(b)<=deg(a) and that b!=0
      const uint64_t deg_a = degree(a);
      const uint64_t deg_b = degree(b);
      const uint64_t Q_N = deg_a - deg_b + 1;

      Q.allocate(Q_N, State::Coefficients);
      // TODO Yuval: Can do better in terms of memory allocation? deg(R) <= deg(b) by definition but it starts as a.
      R.allocate(a_N, State::Coefficients, false /*= memset_zeros*/);

      auto [R_coeffs, __] = R.get_coefficients();
      // init: Q=0, R=a
      cudaMemcpy(R_coeffs, a_coeffs, a_N * sizeof(C), cudaMemcpyDeviceToDevice);

      // TODO Yuval: divide on GPU! no need to copy to host and divide there
      const C lc_b = get_coefficient_on_host(b, deg_b - 1); // largest coeff of b

      // divide and subtract until degree of r is smaller than degree of b
      uint64_t deg_r = degree(R);
      while (deg_r >= deg_b) {
        C lc_r = get_coefficient_on_host(R, deg_r - 1);
        C s_coeff = lc_r * C::inverse(lc_b); // lc_r / lc_b
        uint64_t s_monomial =
          deg_r - deg_b; // divide largest coeff. This is the coeff of 'deg_r-deg_b'. s_monomial=1 is 'x', 2 is x^2 etc.
        add_monomial_inplace(Q, s_coeff, s_monomial); // q = q+x^(degr-degb)
        // TODO Yuval: revisit (#blocks, #threads)
        const int NOF_THREADS = 32;
        const int NOF_BLOCKS = (deg_r + NOF_THREADS - 1) / NOF_THREADS;
        SchoolBookDivisionStepOnR<<<NOF_BLOCKS, NOF_THREADS>>>(
          R_coeffs, b_coeffs, deg_r, deg_b, s_monomial, s_coeff); // r=r-sb
        deg_r = degree(R);
      }
    }

    void quotient(PolyContext& Q, PolyContext& op_a, PolyContext& op_b) override
    {
      // TODO Yuval: implement more efficiently
      GPUPolynomialContext<C, D, I> R = {};
      divide(Q, R, op_a, op_b);
    }

    void remainder(PolyContext& R, PolyContext& op_a, PolyContext& op_b) override
    {
      // TODO Yuval: implement more efficiently
      GPUPolynomialContext<C, D, I> Q = {};
      divide(Q, R, op_a, op_b);
    }

    void divide_by_vanishing_polynomial(PolyContext& out, PolyContext& op_a, uint64_t vanishing_poly_degree) override
    {
      throw std::runtime_error("not implemented yet");
    }

    // arithmetic with monomials
    void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) override
    {
      const uint64_t new_nof_elements = max(poly.get_nof_elements(), monomial + 1);
      poly.transform_to_coefficients(new_nof_elements);
      auto [coeffs, _] = poly.get_coefficients();
      AddSingleElementInplace<<<1, 1>>>(coeffs + monomial, monomial_coeff);
    }

    void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) override
    {
      add_monomial_inplace(poly, C::zero() - monomial_coeff, monomial);
    }

    // // dot product with coefficients
    // ECpoint dot_product_with_coefficients(PolyContext& op, ECpoint* points, uint64_t nof_points) override
    // {
    //   return ECpoint::zero();
    // }

    int32_t degree(PolyContext& p) override
    {
      auto [coeff, nof_coeff] = p.get_coefficients();

      int32_t* d_degree;
      int32_t h_degree;
      cudaMalloc(&d_degree, sizeof(int32_t));
      cudaMemset(d_degree, -1, sizeof(int32_t));
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

    void evaluate(PolyContext& p, const D* domain_x, uint64_t nof_domain_points, I* evaluations /*OUT*/) override
    {
      // TODO Yuval: implement more efficiently
      for (uint64_t i = 0; i < nof_domain_points; ++i) {
        evaluations[i] = evaluate(p, domain_x[i]);
      }
    }

    int64_t
    get_coefficients_on_host(PolyContext& op, C* host_coeffs, int64_t start_idx = 0, int64_t end_idx = -1) override
    {
      const uint64_t nof_coeffs = op.get_nof_elements();
      if (nullptr == host_coeffs) { return nof_coeffs; } // no allocated memory

      end_idx = (end_idx == -1) ? nof_coeffs - 1 : end_idx;

      const bool is_valid_start_idx = start_idx < nof_coeffs && start_idx >= 0;
      const bool is_valid_end_idx = end_idx < nof_coeffs && end_idx >= 0 && end_idx >= start_idx;
      const bool is_valid_indices = is_valid_start_idx && is_valid_end_idx;
      if (!is_valid_indices) {
        // return -1 instead? I could but 'get_coefficient_on_host()' cannot with its current declaration
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "get_coefficients_on_host() invalid indices");
      }

      op.transform_to_coefficients();
      auto [device_coeffs, _] = op.get_coefficients();
      const size_t nof_coeffs_to_copy = end_idx - start_idx + 1;
      cudaMemcpy(host_coeffs, device_coeffs + start_idx, nof_coeffs_to_copy * sizeof(C), cudaMemcpyDeviceToHost);
      return nof_coeffs_to_copy;
    }

    // read coefficients to host
    C get_coefficient_on_host(PolyContext& op, uint64_t coeff_idx) override
    {
      C host_coeff;
      get_coefficients_on_host(op, &host_coeff, coeff_idx, coeff_idx);
      return host_coeff;
    }
  };
} // namespace polynomials