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

    // TODO Yuval: add device context to be able to know which device it is and get a corresponding stream
  public:
    C* init_from_coefficients(uint32_t nof_coefficients, const C* h_coefficients) override
    {
      void* storage = getOrAllocate(nof_coefficients);
      if (h_coefficients) { cudaMemcpy(storage, h_coefficients, nof_coefficients * sizeof(C), cudaMemcpyHostToDevice); }
      set_state(State::Coefficients);
      return m_coefficients;
    }

    I* init_from_rou_evaluations(uint32_t nof_evaluations, const I* h_evaluations) override
    {
      void* storage = getOrAllocate(nof_evaluations);
      if (h_evaluations) { cudaMemcpy(storage, h_evaluations, nof_evaluations * sizeof(I), cudaMemcpyHostToDevice); }
      set_state(State::EvaluationsOnRou);
      return m_evaluations;
    }

    std::pair<C*, uint32_t> get_coefficients() override
    {
      transform_state(State::Coefficients);
      return std::make_pair(m_coefficients, m_nof_elements);
    }
    std::pair<I*, uint32_t> get_rou_evaluations() override
    {
      transform_state(State::EvaluationsOnRou);
      return std::make_pair(m_evaluations, m_nof_elements);
    }

    void print(std::ostream& os) override
    {
      os << "(id=" << PolyContext::m_id << ")[";
      if (m_state == State::Coefficients) {
        for (size_t i = 0; i < m_nof_elements; ++i) {
          os << m_coefficients[i];
          if (i < m_nof_elements - 1) { os << ", "; }
        }
        os << "] (state=coefficients)" << std::endl;
      } else if (m_state == State::EvaluationsOnRou) {
        for (size_t i = 0; i < m_nof_elements; ++i) {
          os << m_evaluations[i];
          if (i < m_nof_elements - 1) { os << ", "; }
        }
        os << "] (state=rou evaluations)" << std::endl;
      }
    }

    ~GPUPolynomialContext() { release(); }

  private:
    enum State { Coefficients, EvaluationsOnRou };
    void release()
    {
      if (m_storage != nullptr) { cudaFree(m_storage); }

      m_storage = nullptr;
      m_coefficients = nullptr;
      m_evaluations = nullptr;
      m_nof_elements = 0;
    }

    void* getOrAllocate(uint32_t nof_elements)
    {
      const uint32_t nof_elements_nearset_power_of_two = 1 << uint32_t(ceil(log2(nof_elements)));
      const bool is_already_allocated = nof_elements_nearset_power_of_two <= m_nof_elements;
      if (is_already_allocated) { return m_storage; }

      std::cout << "GPUPolynomialBackend allocating memory" << std::endl;

      release();
      const uint32_t mem_size = nof_elements_nearset_power_of_two * ElementSize;
      cudaMallocManaged(&m_storage, mem_size); // TODO Yuval use streams and async
      // TODO Yuval: memset zeros?
      m_nof_elements = nof_elements_nearset_power_of_two;
      return m_storage;
    }

    void set_state(State state)
    {
      m_coefficients = nullptr;
      m_evaluations = nullptr;
      switch (state) {
      case State::Coefficients:
        m_coefficients = reinterpret_cast<C*>(m_storage);
        break;
      case State::EvaluationsOnRou:
        m_evaluations = reinterpret_cast<I*>(m_storage);
        break;
      default:
        throw std::runtime_error("not implemented");
      }
      m_state = state;
    }

    void transform_state(State state)
    {
      const bool is_already_in_state = state == m_state;
      if (is_already_in_state) return;

      const bool is_coeff_to_rou_evaluations = state == State::EvaluationsOnRou;

      auto ntt_config = ntt::DefaultNTTConfig<C>();
      if (is_coeff_to_rou_evaluations) {
        ntt::NTT(m_coefficients, m_nof_elements, ntt::NTTDir::kForward, ntt_config, m_evaluations);
      } else { // rou_evalutions to coefficients
        ntt::NTT(m_evaluations, m_nof_elements, ntt::NTTDir::kInverse, ntt_config, m_coefficients);
      }
      set_state(state);
    }

    size_t size() const { return m_nof_elements * ElementSize; }
    State get_state() const { return m_state; }

    // Members
    // ElementSize helps allocate single memory for both coefficients and evaluations
    static constexpr size_t ElementSize = std::max(sizeof(C), sizeof(I));

    uint32_t m_nof_elements;   // #coefficients or #evaluations
    void* m_storage = nullptr; // actual allocated memory. Coefficients/evaluations point to storage
    C* m_coefficients = nullptr;
    I* m_evaluations = nullptr;

    State m_state; // whether data is in coefficients or evaluations form
  };

  /*============================== Polynomial GPU-backend ==============================*/

  template <typename C, typename D, typename I, typename ECpoint>
  class GPUPolynomialBackend : public IPolynomialBackend<C, D, I, ECpoint>
  {
    typedef IPolynomialContext<C, D, I> PolyContext;

  public:
    void add_sub(PolyContext& res, PolyContext& a, PolyContext& b, bool add1_sub0)
    {
      auto [a_coeff_p, a_nof_coeff] = a.get_coefficients();
      auto [b_coeff_p, b_nof_coeff] = b.get_coefficients();

      const auto res_nof_coeff = max(a_nof_coeff, b_nof_coeff);
      auto* res_coeff_p = res.init_from_coefficients(res_nof_coeff);

      const int NOF_THREADS = 32;
      const int NOF_BLOCKS = (res_nof_coeff + NOF_THREADS - 1) / NOF_THREADS;
      AddSubKernel<<<NOF_BLOCKS, NOF_THREADS>>>(a_coeff_p, b_coeff_p, a_nof_coeff, b_nof_coeff, add1_sub0, res_coeff_p);
    }

    void add(PolyContext& res, PolyContext& a, PolyContext& b) override { add_sub(res, a, b, true /*=add*/); }

    void subtract(PolyContext& res, PolyContext& a, PolyContext& b) override { add_sub(res, a, b, false /*=sub*/); }

    void multiply(PolyContext& out, PolyContext& op_a, PolyContext& op_b) override {}
    void divide(PolyContext& Quotient_out, PolyContext& Remainder_out, PolyContext& op_a, PolyContext& op_b) override {}
    void quotient(PolyContext& out, PolyContext& op_a, PolyContext& op_b) override {}
    void remainder(PolyContext& out, PolyContext& op_a, PolyContext& op_b) override {}
    void divide_by_vanishing_polynomial(PolyContext& out, PolyContext& op_a, uint32_t vanishing_poly_degree) override {}
    void reciprocal(PolyContext& out, PolyContext& op) override {}

    // arithmetic with monomials
    void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint32_t monomial) override {}
    void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint32_t monomial) override {}

    // dot product with coefficients
    ECpoint dot_product_with_coefficients(PolyContext& op, ECpoint* points, uint32_t nof_points) override
    {
      return ECpoint::zero();
    }

    int32_t degree(PolyContext& p) override
    {
      auto [coeff, nof_coeff] = p.get_coefficients();

      // TODO Yuval use streams and async
      int32_t* d_degree;
      int32_t h_degree;
      cudaMalloc(&d_degree, sizeof(int32_t));
      // TODO parallelize kernel
      HighestNonZeroIdx<<<1, 1>>>(coeff, nof_coeff, d_degree);
      cudaMemcpy(&h_degree, d_degree, sizeof(int32_t), cudaMemcpyDeviceToHost);
      cudaFree(d_degree);

      return h_degree + 1;
    }

    I evaluate(PolyContext& self, const D& domain_x) override
    {
      // TODO Yuval
      I im = {};
      return im;
    }
    void evaluate(PolyContext& self, const D* domain_x, uint32_t nof_domain_points, I* evaluations /*OUT*/) override {}

    C get_coefficient(PolyContext& op, uint32_t coeff_idx) override
    {
      // TODO Yuval: implement by copying to hostreturn
      return C::zero();
    }
    uint32_t get_coefficients(PolyContext& op, C* coefficients) override
    {
      return 0; // TODO Yuval: implement by copying to hostreturn }
    }
  };
} // namespace polynomials