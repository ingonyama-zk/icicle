

#include <list>
#include "icicle/runtime.h"
#include "icicle/errors.h"
#include "icicle/ntt.h"
#include "icicle/polynomials/polynomial_context.h"
#include "icicle/vec_ops.h"

namespace icicle {

  template <typename C = scalar_t, typename D = C, typename I = C>
  class DefaultPolynomialBackend : public IPolynomialBackend<C, D, I>
  {
    typedef std::shared_ptr<IPolynomialContext<C, D, I>> PolyContext;
    typedef typename IPolynomialContext<C, D, I>::State State;

    int64_t* d_degree = nullptr; // used to avoid alloc/release every time

  public:
    icicleStreamHandle m_stream;
    DefaultPolynomialBackend(const icicleStreamHandle stream) : m_stream{stream}
    {
      ICICLE_CHECK(icicle_malloc_async((void**)&d_degree, sizeof(int64_t), m_stream));
    }
    ~DefaultPolynomialBackend() noexcept { icicle_free_async(d_degree, m_stream); }

    void from_coefficients(PolyContext p, uint64_t nof_coefficients, const C* coefficients) override
    {
      p->from_coefficients(nof_coefficients, coefficients);
    }

    void from_rou_evaluations(PolyContext p, uint64_t nof_evaluations, const I* evaluations) override
    {
      p->from_rou_evaluations(nof_evaluations, evaluations);
    }

    void clone(PolyContext out, PolyContext in) override { out->clone(*in); }

    template <typename T = C>
    T* get_context_storage_mutable(PolyContext p)
    {
      return static_cast<T*>(IPolynomialBackend<C, D, I>::get_context_storage_mutable(p));
    }

    template <typename T = C>
    const T* get_context_storage_immutable(PolyContext p) const
    {
      return static_cast<const T*>(IPolynomialBackend<C, D, I>::get_context_storage_immutable(p));
    }

    void slice(PolyContext out, PolyContext in, uint64_t offset, uint64_t stride, uint64_t size) override
    {
      assert_device_compatability({in});
      auto [in_coeffs, in_size] = in->get_coefficients();
      // size=0 means take as much as elements as there are to take
      uint64_t out_size = (size > 0) ? size : (1 + (in_size - 1 - offset) / stride);

      out->allocate(out_size, State::Coefficients, false /*=memset zeros*/);
      auto out_coeffs = get_context_storage_mutable(out);

      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;

      ICICLE_CHECK(icicle::slice(in_coeffs, offset, stride, out_size, config, out_coeffs));
    }

    void add_sub(PolyContext& res, PolyContext a, PolyContext b, bool add1_sub0)
    {
      assert_device_compatability({a, b});

      // add/sub can be done in both coefficients or evaluations, but operands must be in the same state.
      // For evaluations, same state also means same number of evaluations (and on same domain).
      // If not same state, compute in coefficients since computing in evaluations may require to interpolate a large
      // size. Consider a+b where a is degree 128 and b degree 4. In coefficients b has 4 elements but in evaluations
      // need 128.
      const bool is_same_size = a->get_nof_elements() == b->get_nof_elements();
      const bool is_same_state = a->get_state() == b->get_state();
      const auto output_state = (is_same_size && is_same_state) ? a->get_state() : State::Coefficients;
      const auto output_size = std::max(a->get_nof_elements(), b->get_nof_elements());

      if (State::Coefficients == output_state) {
        a->transform_to_coefficients();
        b->transform_to_coefficients();
      }
      const auto a_mem_p = get_context_storage_immutable(a);
      const auto b_mem_p = get_context_storage_immutable(b);

      res->allocate(output_size, output_state);
      auto res_mem_p = get_context_storage_mutable(res);

      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_b_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;

      const bool a_is_larger_than_b = a->get_nof_elements() > b->get_nof_elements();
      const bool b_is_larger_than_a = a->get_nof_elements() < b->get_nof_elements();

      uint64_t min_op_size = std::min(a->get_nof_elements(), b->get_nof_elements());
      if (add1_sub0) {
        ICICLE_CHECK(icicle::vector_add(a_mem_p, b_mem_p, min_op_size, config, res_mem_p));

        if (a_is_larger_than_b) {
          ICICLE_CHECK(icicle_copy(
            res_mem_p + min_op_size, a_mem_p + min_op_size, sizeof(C) * (a->get_nof_elements() - min_op_size)));
        } else if (b_is_larger_than_a) {
          ICICLE_CHECK(icicle_copy(
            res_mem_p + min_op_size, b_mem_p + min_op_size, sizeof(C) * (b->get_nof_elements() - min_op_size)));
        }
        return;
      }

      // sub case
      ICICLE_CHECK(icicle::vector_sub(a_mem_p, b_mem_p, min_op_size, config, res_mem_p));

      if (a_is_larger_than_b) {
        ICICLE_CHECK(icicle_copy(
          res_mem_p + min_op_size, a_mem_p + min_op_size, sizeof(C) * (a->get_nof_elements() - min_op_size)));
      } else if (b_is_larger_than_a) {
        C zero = C::zero();
        config.is_a_on_device = false;
        ICICLE_CHECK(
          scalar_sub_vec(&zero, b_mem_p + min_op_size, b->get_nof_elements() - min_op_size, config, res_mem_p));
      }
    }

    void add(PolyContext& res, PolyContext a, PolyContext b) override { add_sub(res, a, b, true /*=add*/); }
    void subtract(PolyContext res, PolyContext a, PolyContext b) override { add_sub(res, a, b, false /*=sub*/); }

    void multiply(PolyContext c, PolyContext a, PolyContext b) override
    {
      assert_device_compatability({a, b});

      const bool is_a_scalar = a->get_nof_elements() == 1;
      const bool is_b_scalar = b->get_nof_elements() == 1;

      // TODO: can add kernel that takes the scalar as device memory
      if (is_a_scalar) {
        return multiply(c, b, get_coeff(a, 0));
      } else if (is_b_scalar) {
        return multiply(c, a, get_coeff(b, 0));
      }

      const bool is_multiply_with_cosets = true; // TODO  Yuval: check when faster to do so.
      if (is_multiply_with_cosets) { return multiply_with_cosets(c, a, b); }
      return multiply_with_padding(c, a, b);
    }

    void multiply(PolyContext out, PolyContext p, D scalar) override
    {
      assert_device_compatability({p});

      // element wise multiplication is similar both in coefficients and evaluations (regardless of order too)
      const auto state = p->get_state();
      const auto N = p->get_nof_elements();

      auto p_elements_p =
        state == State::Coefficients ? get_context_storage_immutable<C>(p) : get_context_storage_immutable<I>(p);

      out->allocate(N, state, false /*=memset zeros*/);
      auto out_evals_p =
        state == State::Coefficients ? get_context_storage_mutable<C>(out) : get_context_storage_mutable<I>(out);

      auto config = default_vec_ops_config();
      config.is_a_on_device = false;
      config.is_b_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;
      icicle::scalar_mul_vec(&scalar, p_elements_p, N, config, out_evals_p);
    }

    void multiply_with_padding(PolyContext c, PolyContext a, PolyContext b)
    {
      // TODO Yuval: by using the degree I can optimize the memory size and avoid redundant computations too
      const uint64_t a_N_orig = a->get_nof_elements();
      const uint64_t b_N_orig = b->get_nof_elements();
      const uint64_t N = std::max(a_N_orig, b_N_orig);
      const uint64_t c_N = 2 * N;

      // (1) transform a,b to 2N evaluations
      a->transform_to_evaluations(c_N, true /*=reversed*/);
      b->transform_to_evaluations(c_N, true /*=reversed*/);
      auto [a_evals_p, a_N] = a->get_rou_evaluations();
      auto [b_evals_p, b_N] = b->get_rou_evaluations();

      // (2) allocate c (c=a*b) and compute element-wise multiplication on evaluations
      c->allocate(c_N, State::EvaluationsOnRou_Reversed, false /*=memset zeros*/);
      auto c_evals_p = get_context_storage_mutable<I>(c);

      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_b_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;
      ICICLE_CHECK(icicle::vector_mul(a_evals_p, b_evals_p, c_N, config, c_evals_p));
    }

    void multiply_with_cosets(PolyContext c, PolyContext a, PolyContext b)
    {
      const uint64_t a_N = a->get_nof_elements();
      const uint64_t b_N = b->get_nof_elements();
      const uint64_t N = std::max(a_N, b_N);

      // (1) transform a,b to coefficients such that both have N coefficients
      a->transform_to_coefficients(N);
      b->transform_to_coefficients(N);
      auto [a_coeff_p, _] = a->get_coefficients();
      auto [b_coeff_p, __] = b->get_coefficients();
      // (2) allocate c (c=a*b)
      const uint64_t c_N = 2 * N;
      c->allocate(c_N, State::EvaluationsOnRou_Reversed, false /*=memset zeros*/);
      auto c_evals_low_p = get_context_storage_mutable<I>(c);
      I* c_evals_high_p = c_evals_low_p + N;

      // (3) compute NTT of a,b on coset and write to c
      auto ntt_config = default_ntt_config<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kNR;
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(c_N), &ntt_config.coset_gen));

      ICICLE_CHECK(ntt(a_coeff_p, N, NTTDir::kForward, ntt_config, c_evals_low_p));  // a_H1
      ICICLE_CHECK(ntt(b_coeff_p, N, NTTDir::kForward, ntt_config, c_evals_high_p)); // b_H1

      // (4) compute a_H1 * b_H1 inplace
      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_b_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;
      ICICLE_CHECK(icicle::vector_mul(c_evals_low_p, c_evals_high_p, N, config, c_evals_high_p));

      // (5) transform a,b to evaluations
      a->transform_to_evaluations(N, true /*=reversed*/);
      b->transform_to_evaluations(N, true /*=reversed*/);
      auto [a_evals_p, a_nof_evals] = a->get_rou_evaluations();
      auto [b_evals_p, b_nof_evals] = b->get_rou_evaluations();

      // (6) compute a_H0 * b_H0
      ICICLE_CHECK(icicle::vector_mul(a_evals_p, b_evals_p, N, config, c_evals_low_p));
    }

    void divide(PolyContext Q /*OUT*/, PolyContext R /*OUT*/, PolyContext a, PolyContext b) override
    {
      assert_device_compatability({a, b});

      auto [a_coeffs, a_N] = a->get_coefficients();
      auto [b_coeffs, b_N] = b->get_coefficients();

      const int64_t deg_a = degree(a);
      const int64_t deg_b = degree(b);
      ICICLE_ASSERT(deg_b >= 0) << "Polynomial division:  divide by zero polynomial";

      // init: Q=0, R=a
      Q->allocate(deg_a - deg_b + 1, State::Coefficients, true /*=memset zeros*/);
      auto Q_coeffs = get_context_storage_mutable(Q);

      // TODO Yuval: Can do better in terms of memory allocation? deg(R) <= deg(b) by definition but it starts from
      // deg(a)
      R->allocate(a_N, State::Coefficients, false /*=memset_zeros*/);
      auto R_coeffs = get_context_storage_mutable(R);

      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_b_on_device = true;
      config.is_async = true;
      config.stream = m_stream;
      config.is_result_on_device = true;

      ICICLE_CHECK(icicle::polynomial_division(
        a_coeffs, deg_a, b_coeffs, deg_b, config, Q_coeffs, deg_a - deg_b + 1, R_coeffs, a_N));
    }

    void quotient(PolyContext Q, PolyContext op_a, PolyContext op_b) override
    {
      // TODO: can implement more efficiently?
      auto R = std::make_shared<DefaultPolynomialContext<C, D, I>>(m_stream);
      divide(Q, R, op_a, op_b);
    }

    void remainder(PolyContext R, PolyContext op_a, PolyContext op_b) override
    {
      // TODO: can implement more efficiently?
      auto Q = std::make_shared<DefaultPolynomialContext<C, D, I>>(m_stream);
      divide(Q, R, op_a, op_b);
    }

    void divide_by_vanishing_polynomial(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree) override
    {
      assert_device_compatability({numerator});

      // vanishing polynomial of degree N is the polynomial V(x) such that V(r)=0 for r Nth root-of-unity.
      // For example for N=4 it vanishes on the group [1,W,W^2,W^3] where W is the 4th root of unity. In that
      // case V(x)=(x-1)(x-w)(x-w^2)(x-w^3). It can be easily shown that V(x)=x^N-1. This holds since x^N=1 on this
      // domain (since x is the Nth root of unity).

      // Note that we always represent polynomials with N elements for N a power of two. This is required for NTTs.
      // In addition we consider deg(P) to be this number of elements N even though the real degree may be lower. for
      // example 1+x-2x^2 is degree 2 but we store 4 elements and consider it degree 3.

      // when dividing a polynomial  P(x)/V(x) (The vanishing polynomial) the output is of degree deg(P)-deg(V). There
      // are three cases where V(x) divides P(x) (this is assumed since otherwise the output polynomial does not
      // exist!):
      // (1) deg(P)=2*deg(V): in that case deg(P/V)=deg(V)=N. This is an efficient case since on a domain of size N,
      // the vanishing polynomial evaluates to a constant value. (2) deg(P)=deg(V)=N: in that case the output is a
      // degree 0 polynomial. polynomial (i.e. scalar). (3) general case: deg(P)>2*deg(V): in that case deg(P) is a
      // least 4*deg(V) since N is a power of two. This means that deg(P/V)=deg(P). For example deg(P)=16, deg(V)=4
      // --> deg(P/V)=12 ceiled to 16.

      // When computing we want to divide P(x)'s evals by V(x)'s evals. Since V(x)=0 on this domain we have to compute
      // on a coset.
      // for case (3) we must evaluate V(x) on deg(P) domain size and compute elementwise division on a coset.
      // case (1) is more efficient because we need N evaluations of V(x) on a coset. Note that V(x)=constant on a
      // coset of rou. This is because V(wu)=(wu)^N-1=W^N*u^N-1 = 1*u^N-1 (as w^N=1 for w Nth root of unity). case (2)
      // can be computed like case (1).

      const bool is_case_2N = numerator->get_nof_elements() == 2 * vanishing_poly_degree;
      const bool is_case_N = numerator->get_nof_elements() == vanishing_poly_degree;
      if (is_case_2N) {
        divide_by_vanishing_case_2N(out, numerator, vanishing_poly_degree);
      } else if (is_case_N) {
        divide_by_vanishing_case_N(out, numerator, vanishing_poly_degree);
      } else {
        divide_by_vanishing_general_case(out, numerator, vanishing_poly_degree);
      }
    }

    void divide_by_vanishing_general_case(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree)
    {
      // General case: P(x)/V(x) where v is of degree N and p of any degree>N

      // (1) allocate vanishing polynomial in coefficients form
      // TODO Yuval: maybe instead of taking numerator memory and modiyfing it diretcly add a state for evaluations
      // on coset of rou. In that case I can remain in this state and also won't need to access input memory
      // directly
      numerator->transform_to_coefficients();
      auto numerator_coeffs = get_context_storage_mutable(numerator);
      const auto N = numerator->get_nof_elements();
      ICICLE_ASSERT(vanishing_poly_degree <= N) << "divide_by_vanishing_polynomial(): degree is too large";

      out->allocate(N, State::Coefficients, true /*=set zeros*/);
      add_monomial_inplace(out, C::zero() - C::one(), 0);         //-1
      add_monomial_inplace(out, C::one(), vanishing_poly_degree); //+x^n

      // (2) NTT on coset. Note that NTT on ROU evaluates to zeros for vanihsing polynomials by definition.
      // Therefore evaluation on coset is required to compute non-zero evaluations, which make element-wise division
      // possible
      auto out_coeffs = get_context_storage_mutable(out);
      auto ntt_config = default_ntt_config<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kNM;
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(2 * N), &ntt_config.coset_gen));

      ICICLE_CHECK(ntt(out_coeffs, N, NTTDir::kForward, ntt_config, out_coeffs));
      ICICLE_CHECK(ntt(numerator_coeffs, N, NTTDir::kForward, ntt_config, numerator_coeffs));

      // (3) element wise division
      auto config = default_vec_ops_config();
      config.is_a_on_device = false;
      config.is_b_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;
      ICICLE_CHECK(icicle::vector_div(numerator_coeffs, out_coeffs, N, config, out_coeffs));

      // (4) INTT back both numerator and out
      ntt_config.ordering = Ordering::kMN;
      ICICLE_CHECK(ntt(out_coeffs, N, NTTDir::kInverse, ntt_config, out_coeffs));
      ICICLE_CHECK(ntt(numerator_coeffs, N, NTTDir::kInverse, ntt_config, numerator_coeffs));
    }

    void divide_by_vanishing_case_2N(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree)
    {
      // in that special case the numertaor has 2N elements and output will be N elements
      ICICLE_ASSERT(numerator->get_nof_elements() == 2 * vanishing_poly_degree)
        << "invalid input size. Expecting numerator to be of size 2N";

      // In the case where deg(P)=2N, I can transform numerator to Reversed-evals -> The second half is
      // a reversed-coset of size N with coset-gen the 2N-th root of unity.
      const int N = vanishing_poly_degree;
      numerator->transform_to_evaluations(2 * N, true /*=reversed*/);
      // allocate output in coeffs because it will be calculated on a coset but I don't have such a state so will have
      // to INTT back to coeffs
      auto numerator_evals_reversed_p = get_context_storage_immutable<I>(numerator);
      out->allocate(N, State::Coefficients, false /*=set zeros*/);
      auto out_evals_reversed_p = get_context_storage_mutable<I>(out);

      auto ntt_config = default_ntt_config<C>();
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(2 * N), &ntt_config.coset_gen));
      // compute inv(u^N-1);
      D v_coset_eval = D::inverse(D::pow(ntt_config.coset_gen, N) - D::one());

      auto config = default_vec_ops_config();
      config.is_a_on_device = false;
      config.is_b_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;
      icicle::scalar_mul_vec(
        &v_coset_eval, numerator_evals_reversed_p + N /*second half is the reversed coset*/, N, config,
        out_evals_reversed_p);

      // INTT back from reversed evals on coset to coeffs
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kRN;
      ntt(out_evals_reversed_p, N, NTTDir::kInverse, ntt_config, out_evals_reversed_p);
    }

    void divide_by_vanishing_case_N(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree)
    {
      // in that special case the numertaor has N elements and output will be N elements
      ICICLE_ASSERT(numerator->get_nof_elements() == vanishing_poly_degree)
        << "invalid input size. Expecting numerator to be of size N";

      const int N = vanishing_poly_degree;
      numerator->transform_to_coefficients(N);
      auto numerator_evals_reversed_p = get_context_storage_immutable<I>(numerator);
      out->allocate(N, State::Coefficients, false /*=set zeros*/);
      auto out_evals_reversed_p = get_context_storage_mutable<I>(out);

      // (1) NTT numerator to coset evals (directly to out)
      auto ntt_config = default_ntt_config<C>();
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(2 * N), &ntt_config.coset_gen));
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kNM;
      ntt(numerator_evals_reversed_p, N, NTTDir::kForward, ntt_config, out_evals_reversed_p);

      // (2) divide by constant value (that V(x) evaluates to on the coset)
      D v_coset_eval = D::inverse(D::pow(ntt_config.coset_gen, N) - D::one());

      auto config = default_vec_ops_config();
      config.is_a_on_device = false;
      config.is_b_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;
      icicle::scalar_mul_vec(&v_coset_eval, out_evals_reversed_p, N, config, out_evals_reversed_p);

      // (3) INTT back from coset to coeffs
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kMN;
      ntt(out_evals_reversed_p, N, NTTDir::kInverse, ntt_config, out_evals_reversed_p);
    }

    // arithmetic with monomials
    void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) override
    {
      const uint64_t new_nof_elements = std::max(poly->get_nof_elements(), monomial + 1);
      poly->transform_to_coefficients(new_nof_elements);
      auto coeffs = get_context_storage_mutable(poly);

      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_b_on_device = false;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;

      ICICLE_CHECK(icicle::vector_add(coeffs + monomial, &monomial_coeff, 1, config, coeffs + monomial));
    }

    void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) override
    {
      add_monomial_inplace(poly, C::zero() - monomial_coeff, monomial);
    }

    int64_t degree(PolyContext p) override { return degree_internal(p, p->get_nof_elements()); }

    // search degree starting from len, searching down (towards coeff0)
    int64_t degree_internal(PolyContext p, uint64_t len)
    {
      auto [coeff, _] = p->get_coefficients();

      int64_t h_degree;

      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_result_on_device = true;
      config.is_async = true;
      config.stream = m_stream;

      ICICLE_CHECK(icicle::highest_non_zero_idx(coeff, len, config, d_degree));
      // sync copy to make sure return value is copied to host
      ICICLE_CHECK(icicle_copy_async(&h_degree, d_degree, sizeof(int64_t), m_stream));
      ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // sync to make sure return value is copied to host

      return h_degree;
    }

  public:
    void evaluate(PolyContext p, const D* x, I* eval) override { evaluate_on_domain(p, x, 1, eval); }

    void evaluate_on_domain(PolyContext p, const D* domain, uint64_t size, I* evaluations /*OUT*/) override
    {
      auto [coeff, nof_coeff] = p->get_coefficients();

      auto config = default_vec_ops_config();
      config.is_a_on_device = true;
      config.is_b_on_device = !is_host_ptr(domain);
      config.is_result_on_device = !is_host_ptr(evaluations);
      config.is_async = true;
      config.stream = m_stream;
      ICICLE_CHECK(icicle::polynomial_eval(coeff, nof_coeff, domain, size, config, evaluations));
    }

    void evaluate_on_rou_domain(PolyContext p, uint64_t domain_log_size, I* evals /*OUT*/) override
    {
      const uint64_t poly_size = p->get_nof_elements();
      const uint64_t domain_size = 1 << domain_log_size;
      const bool is_evals_on_host = is_host_ptr(evals);

      I* d_evals = evals;
      // if evals on host, allocate memory
      if (is_evals_on_host) { ICICLE_CHECK(icicle_malloc_async((void**)&d_evals, domain_size * sizeof(I), m_stream)); }

      // If domain size is smaller the polynomial size -> transform to evals and copy the evals with stride.
      // Else, if in coeffs copy coeffs to evals mem and NTT inplace to compute the evals, else INTT to d_evals and
      // back inplace to larger domain
      const bool is_domain_size_smaller_than_poly_size = domain_size <= poly_size;
      if (is_domain_size_smaller_than_poly_size) {
        // TODO Yuval: in reversed evals, can reverse the first 'domain_size' elements to d_evals instead of
        // transforming back to evals.
        p->transform_to_evaluations();
        const auto stride = poly_size / domain_size;

        auto config = default_vec_ops_config();
        config.is_a_on_device = true;
        config.is_result_on_device = true;
        config.is_async = true;
        config.stream = m_stream;
        ICICLE_CHECK(
          icicle::slice(get_context_storage_immutable<I>(p), 0 /*offset*/, stride, domain_size, config, d_evals));
      } else {
        ICICLE_CHECK(icicle_memset(d_evals, 0, domain_size * sizeof(I)));
        auto ntt_config = default_ntt_config<D>();
        ntt_config.are_inputs_on_device = true;
        ntt_config.are_outputs_on_device = true;
        ntt_config.is_async = true;
        // TODO Yuval: in evals I can NTT directly to d_evals without changing my state
        switch (p->get_state()) {
        case State::Coefficients: {
          // copy to evals memory and inplace NTT of domain size
          ICICLE_CHECK(icicle_copy(d_evals, get_context_storage_immutable<I>(p), poly_size * sizeof(I)));
          ntt_config.ordering = Ordering::kNN;
          ntt(d_evals, domain_size, NTTDir::kForward, ntt_config, d_evals);
        } break;
        case State::EvaluationsOnRou_Natural:
        case State::EvaluationsOnRou_Reversed: {
          const bool is_from_natrual = p->get_state() == State::EvaluationsOnRou_Natural;
          // INTT to coeffs and back to evals
          ntt_config.ordering = is_from_natrual ? Ordering::kNM : Ordering::kRN;
          ntt(get_context_storage_immutable<I>(p), poly_size, NTTDir::kInverse, ntt_config, d_evals);
          ntt_config.ordering = is_from_natrual ? Ordering::kMN : Ordering::kNN;
          ntt(d_evals, poly_size, NTTDir::kForward, ntt_config, d_evals);
        } break;
        default:
          ICICLE_ASSERT(false) << "Invalid state to compute evaluations";
          break;
        }
      }

      // release memory if allocated
      if (is_evals_on_host) {
        ICICLE_CHECK(icicle_copy_async(evals, d_evals, domain_size * sizeof(I), m_stream));
        ICICLE_CHECK(icicle_free_async(d_evals, m_stream));
      }

      // sync since user cannot reuse this stream so need to make sure evals are computed
      ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // sync to make sure return value is copied to host
    }

    uint64_t copy_coeffs(PolyContext op, C* out_coeffs, uint64_t start_idx, uint64_t end_idx) override
    {
      const uint64_t nof_coeffs = op->get_nof_elements();
      if (nullptr == out_coeffs) { return nof_coeffs; } // no allocated memory

      const bool is_valid_start_idx = start_idx < nof_coeffs;
      const bool is_valid_end_idx = end_idx < nof_coeffs && end_idx >= start_idx;
      const bool is_valid_indices = is_valid_start_idx && is_valid_end_idx;
      if (!is_valid_indices) {
        // return -1 instead? I could but 'get_coeff()' cannot with its current declaration
        ICICLE_ASSERT(false) << "copy_coeffs() invalid indices";
      }

      op->transform_to_coefficients();
      auto [device_coeffs, _] = op->get_coefficients();
      const size_t nof_coeffs_to_copy = end_idx - start_idx + 1;
      ICICLE_CHECK(icicle_copy_async(out_coeffs, device_coeffs + start_idx, nof_coeffs_to_copy * sizeof(C), m_stream));
      ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // sync to make sure return value is copied

      return nof_coeffs_to_copy;
    }

    // read coefficients to host
    C get_coeff(PolyContext op, uint64_t coeff_idx) override
    {
      C host_coeff;
      copy_coeffs(op, &host_coeff, coeff_idx, coeff_idx);
      return host_coeff;
    }

    std::tuple<IntegrityPointer<C>, uint64_t /*size*/> get_coefficients_view(PolyContext p) override
    {
      return p->get_coefficients_view();
    }

    inline void assert_device_compatability(const std::list<PolyContext>& polys) const
    {
      for (const PolyContext& p : polys) {
        ICICLE_CHECK(icicle_is_active_device_memory(get_context_storage_immutable(p)));
      }
    }
  };

} // namespace icicle