#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include <array>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <utility>

// IntegerRingRns: Implements integer-rings in RNS (Residue Number System) representation.
// In RNS, an integer is represented as a tuple of integers, each modulo a different field prime.
// Given q = p1 * p2 * ... * pn, we represent x as (x mod p1, x mod p2, ..., x mod pn).
// This class provides arithmetic operations on integers represented in RNS.
template <typename RNS_CONFIG>
class IntegerRingRns
{
public:
  static constexpr unsigned limbs_count = RNS_CONFIG::limbs_count;
  using Fields = typename RNS_CONFIG::Fields;
  using Zq = typename RNS_CONFIG::Zq;
  static constexpr unsigned nof_fields = std::tuple_size_v<Fields>;
  static constexpr std::array<unsigned, limbs_count> field_offset = RNS_CONFIG::field_offset;

  template <size_t I>
  using field_type = std::tuple_element_t<I, Fields>;

  storage<RNS_CONFIG::limbs_count> limbs_storage;

  friend std::ostream& operator<<(std::ostream& os, const IntegerRingRns& x)
  {
    x.print(os, std::make_index_sequence<nof_fields>{});
    return os;
  }

  template <size_t... I>
  void print(std::ostream& os, std::index_sequence<I...>) const
  {
    // Print the fields element wise
    ((os << " [" << I << "]=" << *get_field<I>()), ...);
  }

  // Access Field Elements
  template <size_t I>
  HOST_DEVICE_INLINE field_type<I>* get_field()
  {
    return reinterpret_cast<field_type<I>*>(limbs_storage.limbs + field_offset[I]);
  }

  template <size_t I>
  HOST_DEVICE_INLINE const field_type<I>* get_field() const
  {
    return reinterpret_cast<const field_type<I>*>(limbs_storage.limbs + field_offset[I]);
  }

  // Apply operations
  template <typename Op, size_t... I>
  static HOST_DEVICE_INLINE IntegerRingRns
  apply_binary_op(const IntegerRingRns& a, const IntegerRingRns& b, Op op, std::index_sequence<I...>)
  {
    IntegerRingRns result;
    ((*(result.template get_field<I>()) = op(*(a.template get_field<I>()), *(b.template get_field<I>()))), ...);
    return result;
  }

  template <typename Op, size_t... I>
  static HOST_DEVICE_INLINE IntegerRingRns apply_op_unary(const IntegerRingRns& x, Op op, std::index_sequence<I...>)
  {
    IntegerRingRns result;
    ((*(result.template get_field<I>()) = op(*(x.template get_field<I>()))), ...);
    return result;
  }

  template <typename Op, size_t... I>
  static HOST_DEVICE_INLINE IntegerRingRns apply_op_inplace(IntegerRingRns& x, Op op, std::index_sequence<I...>)
  {
    ((*(x.template get_field<I>()) = op(*(x.template get_field<I>()))), ...);
    return x;
  }

  // Constructors
  IntegerRingRns() = default;

  template <typename Zq>
  IntegerRingRns(const Zq& zq)
  {
    convert_direct_to_rns(&zq.limbs_storage, &limbs_storage);
  }

  // Identity Elements
  static HOST_DEVICE_INLINE IntegerRingRns zero()
  {
    IntegerRingRns zero;
    return apply_op_inplace(zero, [](auto x) { return x.zero(); }, std::make_index_sequence<nof_fields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns one()
  {
    IntegerRingRns one;
    return apply_op_inplace(one, [](auto x) { return x.one(); }, std::make_index_sequence<nof_fields>{});
  }

  // Random Elements
  static HOST_INLINE IntegerRingRns rand_host()
  {
    IntegerRingRns rand_element;
    return apply_op_inplace(
      rand_element, [](auto rns_element) { return rns_element.rand_host(); }, std::make_index_sequence<nof_fields>{});
  }

  static void rand_host_many(IntegerRingRns* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  static HOST_DEVICE_INLINE IntegerRingRns from(uint32_t value)
  {
    IntegerRingRns u32;
    return apply_op_inplace(u32, [value](auto x) { return x.from(value); }, std::make_index_sequence<nof_fields>{});
  }

  template <typename T>
  static constexpr bool has_member_omegas_count()
  {
    return sizeof(T::omegas_count) > 0;
  }

  // Arithmetic Operators
  friend HOST_DEVICE_INLINE IntegerRingRns operator+(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x + y; }, std::make_index_sequence<nof_fields>{});
  }

  friend HOST_DEVICE_INLINE IntegerRingRns operator-(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x - y; }, std::make_index_sequence<nof_fields>{});
  }

  friend HOST_DEVICE_INLINE IntegerRingRns operator*(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x * y; }, std::make_index_sequence<nof_fields>{});
  }

  // Equality Operators
  friend HOST_DEVICE bool operator==(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return icicle_math::template is_equal<limbs_count>(a.limbs_storage, b.limbs_storage);
  }

  friend HOST_DEVICE bool operator!=(const IntegerRingRns& a, const IntegerRingRns& b) { return !(a == b); }

  static HOST_DEVICE_INLINE IntegerRingRns neg(const IntegerRingRns& x)
  {
    return apply_op_unary(x, [](auto x) { return x.neg(x); }, std::make_index_sequence<nof_fields>{});
  }

  // Additional ops
  static HOST_DEVICE_INLINE IntegerRingRns inverse(const IntegerRingRns& x)
  {
    // Note that if we don't return zero then it's not consistent with the direct Zq type.
    // Also note that has_inverse() in RNS is very cheap (simply checks all elements are non-zero)
    if (!has_inverse(x)) { return zero(); }
    return apply_op_unary(x, [](auto x) { return x.inverse(x); }, std::make_index_sequence<nof_fields>{});
  }

  template <size_t... I>
  static HOST_DEVICE auto has_inverse_impl(const IntegerRingRns& x, std::index_sequence<I...>)
  {
    // compute element-wise is_zero for each field
    const auto element_wise_is_zero = std::make_tuple((*(x.template get_field<I>()) != field_type<I>::zero())...);
    // Reduce tuple to a single bool (true if all elements are nonzero)
    bool all_nonzero = (... && std::get<I>(element_wise_is_zero));
    return all_nonzero;
  }

  static HOST_DEVICE bool has_inverse(const IntegerRingRns& x)
  {
    return has_inverse_impl(x, std::make_index_sequence<nof_fields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns pow(const IntegerRingRns& x, const IntegerRingRns& y)
  {
    return apply_binary_op(x, y, [](auto x, auto y) { return x.pow(x, y); }, std::make_index_sequence<nof_fields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns to_montgomery(const IntegerRingRns& x)
  {
    return apply_op_unary(x, [](auto x) { return x.to_montgomery(x); }, std::make_index_sequence<nof_fields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns from_montgomery(const IntegerRingRns& x)
  {
    return apply_op_unary(x, [](auto x) { return x.from_montgomery(x); }, std::make_index_sequence<nof_fields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns sqr(const IntegerRingRns& x) { return x * x; }

  static HOST_DEVICE_INLINE IntegerRingRns inv_log_size(uint32_t logn)
  {
    IntegerRingRns res;
    return apply_op_inplace(
      res, [logn](auto x) { return decltype(x)::inv_log_size(logn); }, std::make_index_sequence<nof_fields>{});
  }

  // Roots of unity
  static constexpr HOST_INLINE unsigned get_omegas_count()
  {
    if constexpr (has_member_omegas_count<RNS_CONFIG>()) {
      return RNS_CONFIG::omegas_count;
    } else {
      return 0;
    }
  }

  static HOST_DEVICE_INLINE IntegerRingRns omega(uint32_t logn)
  {
    IntegerRingRns res;
    return apply_op_inplace(
      res, [logn](auto x) { return decltype(x)::omega(logn); }, std::make_index_sequence<nof_fields>{});
  }

  // Conversion Functions
  static constexpr HOST_DEVICE_INLINE void
  convert_direct_to_rns(const storage<RNS_CONFIG::limbs_count>* zq, storage<RNS_CONFIG::limbs_count>* zqrns /*OUT*/)
  {
    // Reduce zq mod pi, elementwise
    const bool inplace = zq == zqrns;
    IntegerRingRns* rns_out_casted = (IntegerRingRns*)(zqrns);
    if (inplace) {
      // Inplace: copy input/output to local memory (GPU regs or CPU stack) and reduce from there
      IntegerRingRns tmp_local_mem;
      tmp_local_mem.limbs_storage = *zq;
      apply_op_inplace(
        *rns_out_casted,
        [&](auto x) {
          return decltype(x)::from((std::byte*)&tmp_local_mem, RNS_CONFIG::limbs_count * sizeof(uint32_t));
        },
        std::make_index_sequence<nof_fields>{});
    } else {
      // reduce directly from input to output
      apply_op_inplace(
        *rns_out_casted,
        [&](auto x) { return decltype(x)::from((std::byte*)zq, RNS_CONFIG::limbs_count * sizeof(uint32_t)); },
        std::make_index_sequence<nof_fields>{});
    }
  }

  // Conversion to/from direct representation

  static constexpr HOST_DEVICE_INLINE void
  convert_rns_to_direct(const storage<RNS_CONFIG::limbs_count>* zqrns, storage<RNS_CONFIG::limbs_count>* zq /*OUT*/)
  {
    // Note: here we compute the CRT algorithm to convert from RNS to direct representation

    constexpr unsigned sizeof_limb = sizeof(zq->limbs[0]);
    // Note: without those, CUDA cannot compile this code to GPU since those values are not in device memory
    constexpr auto field_offsets = RNS_CONFIG::field_offset;
    constexpr auto field_limb_counts = RNS_CONFIG::field_limb_counts;
    constexpr auto crt_weights = RNS_CONFIG::crt_weights;

    // Compute Σ(Wi * xi)
    Zq zq_sum = Zq::zero();
    // TODO UNROLL?
    for (size_t i = 0; i < nof_fields; i++) {
      Zq xi{};
      // TODO: can we avoid this memcpy? Probably yes if we use a low level multiplier, rather than the Zq multiplier
      std::memcpy(
        &xi, reinterpret_cast<const std::byte*>(zqrns) + field_offsets[i] * sizeof_limb,
        field_limb_counts[i] * sizeof_limb);
      zq_sum = zq_sum + (xi * Zq{crt_weights.storages[i]});
    }

    // Store result
    *zq = zq_sum.limbs_storage;
  }

  Zq to_direct() const
  {
    Zq zq;
    convert_rns_to_direct(&limbs_storage, &zq.limbs_storage);
    return zq;
  }

  template <typename Zq>
  static constexpr HOST_DEVICE IntegerRingRns from_direct(const Zq& zq)
  {
    IntegerRingRns res;
    convert_direct_to_rns(&zq.limbs_storage, &res.limbs_storage);
    return res;
  }
};

template <class CONFIG>
struct std::hash<IntegerRingRns<CONFIG>> {
  std::size_t operator()(const IntegerRingRns<CONFIG>& key) const
  {
    std::size_t hash = 0;
    for (int i = 0; i < CONFIG::limbs_count; i++)
      hash ^= std::hash<uint32_t>()(key.limbs_storage.limbs[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

#ifdef __CUDACC__
template <class CONFIG>
struct SharedMemory<IntegerRingRns<CONFIG>> {
  __device__ IntegerRingRns<CONFIG>* getPointer()
  {
    extern __shared__ IntegerRingRns<CONFIG> s_scalar_rns_[];
    return s_scalar_rns_;
  }
};
#endif // __CUDACC__