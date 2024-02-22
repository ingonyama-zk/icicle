#pragma once

// TODO Yuval: use current device context and stream
#include "cuda_runtime.h"
#include "appUtils/ntt/ntt.cuh"
#include "polynomials/gpu_backend/kernels.cuh"

namespace polynomials {
  template <typename C, typename D, typename I>
  class GPUPolynomialBackend : public IPolynomialBackend<C, D, I>
  {
  public:
    enum State { Coefficients, EvaluationsOnRou };

    ~GPUPolynomialBackend()
    {
      std::cout << "~GPUPolynomialBackend(id=" << IPolynomialBackend<C, D, I>::m_id << ")" << std::endl;
      release();
    }

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

    size_t size() const { return m_nof_elements * ElementSize; }

    void print(std::ostream& os) override
    {
      os << "(id=" << IPolynomialBackend<C, D, I>::m_id << ")[";
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

    void init_from_coefficients(const C* coefficients, uint32_t nof_coefficients) override
    {
      void* storage = getOrAllocate(nof_coefficients);
      cudaMemcpy(storage, coefficients, nof_coefficients * sizeof(C), cudaMemcpyHostToDevice);
      set_state(State::Coefficients);
    }

    void init_from_rou_evaluations(const I* evaluations, uint32_t nof_evaluations) override
    {
      void* storage = getOrAllocate(nof_evaluations);
      cudaMemcpy(storage, evaluations, nof_evaluations * sizeof(I), cudaMemcpyHostToDevice);
      set_state(State::EvaluationsOnRou);
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

    // TODO Yuval: how to avoid this ugly downcast??
    static inline GPUPolynomialBackend& get_backend(const Polynomial<C, D, I>& poly)
    {
      // TODO Yuval: backend object per GPU? then also check same GPU. If yes, can use the stream, else throw error
      auto backend = dynamic_cast<GPUPolynomialBackend<C, D, I>*>(poly.m_backend.get());
      if (nullptr == backend) { throw std::runtime_error("[ERROR] expecting GPUPolynomialBackend"); }
      return *backend;
    }

    void add_sub(Polynomial<C, D, I>& res, const Polynomial<C, D, I>& a, const Polynomial<C, D, I>& b, bool add1_sub0)
    {
      auto& res_backend = get_backend(res);
      auto& a_backend = get_backend(a);
      auto& b_backend = get_backend(b);

      // TODO support computation in evaluations form too
      a_backend.transform_state(State::Coefficients);
      b_backend.transform_state(State::Coefficients);

      res_backend.getOrAllocate(max(a_backend.m_nof_elements, b_backend.m_nof_elements));
      res_backend.set_state(State::Coefficients);

      const int NOF_THREADS = 32;
      const int NOF_BLOCKS = (res_backend.m_nof_elements + NOF_THREADS - 1) / NOF_THREADS;
      AddSubKernel<<<NOF_BLOCKS, NOF_THREADS>>>(
        a_backend.m_coefficients, b_backend.m_coefficients, a_backend.m_nof_elements, b_backend.m_nof_elements,
        add1_sub0, res_backend.m_coefficients);
    }

    void add(Polynomial<C, D, I>& res, const Polynomial<C, D, I>& a, const Polynomial<C, D, I>& b) override
    {
      add_sub(res, a, b, true /*=add*/);
    }

    void subtract(Polynomial<C, D, I>& res, const Polynomial<C, D, I>& a, const Polynomial<C, D, I>& b) override
    {
      add_sub(res, a, b, false /*=sub*/);
    }

    void divide(
      Polynomial<C, D, I>& Quotient,
      Polynomial<C, D, I>& Remainder,
      const Polynomial<C, D, I>& a,
      const Polynomial<C, D, I>& b) override
    {
    }
    void add_monomial_inplace(Polynomial<C, D, I>& self, C monomial_coeff, uint32_t monomial) override {}

    int32_t degree(Polynomial<C, D, I>& p) override
    {
      auto& backend = get_backend(p);
      backend.transform_state(State::Coefficients);

      // TODO Yuval use streams and async
      int32_t* d_degree;
      int32_t h_degree;
      cudaMalloc(&d_degree, sizeof(int32_t));
      // TODO parallelize kernel
      HighestNonZeroIdx<<<1, 1>>>(backend.m_coefficients, backend.m_nof_elements, d_degree);
      cudaMemcpy(&h_degree, d_degree, sizeof(int32_t), cudaMemcpyDeviceToHost);
      cudaFree(d_degree);

      return h_degree + 1;
    }

    I evaluate(Polynomial<C, D, I>& self, const D& domain_x) override
    {
      // TODO Yuval
      I im = {};
      return im;
    }
    void
    evaluate(Polynomial<C, D, I>& self, const D* domain_x, uint32_t nof_domain_points, I* evaluations /*OUT*/) override
    {
    }

    C get_coefficient(Polynomial<C, D, I>& self, uint32_t coeff_idx) override
    {
      // TODO Yuval
      C coeff = {};
      return coeff;
    }
    // if coefficients==nullptr, fills nof_coeff only
    void get_coefficients(Polynomial<C, D, I>& self, C* coefficients, uint32_t& nof_coeff) override
    {
      // TODO Yuval
    }

    // ElementSize helps allocate single memory for both coefficients and evaluations
    static constexpr size_t ElementSize = std::max(sizeof(C), sizeof(I));

    uint32_t m_nof_elements;   // #coefficients or #evaluations
    void* m_storage = nullptr; // actual allocated memory. Coefficients/evaluations point to storage
    C* m_coefficients = nullptr;
    I* m_evaluations = nullptr;

    State m_state; // whether data is in coefficients or evaluations form
  };

} // namespace polynomials