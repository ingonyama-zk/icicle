#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include <array>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <utility>

// This class is a template class that implement integer-rings in RNS representation
// In RNS we represent an integer as a tuple of integers, each in a different field.
// given q=p1*p2*...*pn, we represent an integer x as a tuple (x mod p1, x mod p2, ..., x mod pn)
// The class is parameterized by a configuration class that defines the fields and the number of limbs
// The class provides basic arithmetic operations on the integer ring
template <typename RNS_CONFIG>
class IntegerRingRns
{
public:
  // TODO Yuval: clean this class up a little, add comments, and make sure it is consistent with the rest of the code
  static constexpr unsigned limbs_count = RNS_CONFIG::limbs_count;
  using Fields = typename RNS_CONFIG::Fields;
  static constexpr unsigned NofFields = std::tuple_size_v<Fields>;
  static constexpr std::array<unsigned, limbs_count> FieldOffset = RNS_CONFIG::FieldOffset;
  template <size_t I>
  using FieldType = std::tuple_element_t<I, Fields>;

  storage<RNS_CONFIG::limbs_count> limbs_storage;

  // Stream Output
  friend std::ostream& operator<<(std::ostream& os, const IntegerRingRns& xs)
  {
    std::stringstream hex_string;
    hex_string << std::hex << std::setfill('0');

    for (int i = 0; i < limbs_count; i++) {
      hex_string << std::setw(8) << xs.limbs_storage.limbs[limbs_count - i - 1];
    }

    os << "0x" << hex_string.str();
    return os;
  }

  // `get_field` with correct access to `limbs_storage.limbs`
  template <size_t I>
  HOST_DEVICE_INLINE FieldType<I>* get_field()
  {
    return reinterpret_cast<FieldType<I>*>(limbs_storage.limbs + FieldOffset[I]);
  }

  template <size_t I>
  HOST_DEVICE_INLINE const FieldType<I>* get_field() const
  {
    return reinterpret_cast<const FieldType<I>*>(limbs_storage.limbs + FieldOffset[I]);
  }

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

  static HOST_DEVICE_INLINE IntegerRingRns zero()
  {
    IntegerRingRns zero;
    return apply_op_inplace(zero, [](auto x) { return x.zero(); }, std::make_index_sequence<NofFields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns one()
  {
    IntegerRingRns one;
    return apply_op_inplace(one, [](auto x) { return x.one(); }, std::make_index_sequence<NofFields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns from(uint32_t value)
  {
    IntegerRingRns u32;
    return apply_op_inplace(u32, [value](auto x) { return x.from(value); }, std::make_index_sequence<NofFields>{});
  }

  template <typename T>
  static constexpr bool has_member_omegas_count()
  {
    return sizeof(T::omegas_count) > 0;
  }

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
      res, [logn](auto x) { return decltype(x)::omega(logn); }, std::make_index_sequence<NofFields>{});
  }

  // Operator Overloads
  friend HOST_DEVICE_INLINE IntegerRingRns operator+(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x + y; }, std::make_index_sequence<NofFields>{});
  }

  friend HOST_DEVICE_INLINE IntegerRingRns operator-(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x - y; }, std::make_index_sequence<NofFields>{});
  }

  friend HOST_DEVICE_INLINE IntegerRingRns operator*(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x * y; }, std::make_index_sequence<NofFields>{});
  }

  friend HOST_DEVICE bool operator==(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return icicle_math::template is_equal<limbs_count>(a.limbs_storage, b.limbs_storage);
  }

  friend HOST_DEVICE bool operator!=(const IntegerRingRns& a, const IntegerRingRns& b) { return !(a == b); }

  static HOST_INLINE IntegerRingRns rand_host()
  {
    IntegerRingRns rand_element;
    return apply_op_inplace(
      rand_element, [](auto rns_element) { return rns_element.rand_host(); }, std::make_index_sequence<NofFields>{});
  }

  static void rand_host_many(IntegerRingRns* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  static HOST_DEVICE_INLINE IntegerRingRns neg(const IntegerRingRns& x)
  {
    return apply_op_unary(x, [](auto x) { return x.neg(x); }, std::make_index_sequence<NofFields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns inverse(const IntegerRingRns& x)
  {
    return apply_op_unary(x, [](auto x) { return x.inverse(x); }, std::make_index_sequence<NofFields>{});
  }

  template <size_t... I>
  static HOST_DEVICE auto has_inverse_impl(const IntegerRingRns& x, std::index_sequence<I...>)
  {
    // compute element-wise is_zero for each field
    const auto element_wise_is_zero = std::make_tuple((*(x.template get_field<I>()) != FieldType<I>::zero())...);
    // Reduce tuple to a single bool (true if all elements are nonzero)
    bool all_nonzero = (... && std::get<I>(element_wise_is_zero));
    return all_nonzero;
  }

  static HOST_DEVICE bool has_inverse(const IntegerRingRns& x)
  {
    return has_inverse_impl(x, std::make_index_sequence<NofFields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns pow(const IntegerRingRns& x, const IntegerRingRns& y)
  {
    return apply_binary_op(x, y, [](auto x, auto y) { return x.pow(x, y); }, std::make_index_sequence<NofFields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns to_montgomery(const IntegerRingRns& x)
  {
    return apply_op_unary(x, [](auto x) { return x.to_montgomery(x); }, std::make_index_sequence<NofFields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns from_montgomery(const IntegerRingRns& x)
  {
    return apply_op_unary(x, [](auto x) { return x.from_montgomery(x); }, std::make_index_sequence<NofFields>{});
  }

  static HOST_DEVICE_INLINE IntegerRingRns sqr(const IntegerRingRns& x) { return x * x; }

  static HOST_DEVICE_INLINE IntegerRingRns inv_log_size(uint32_t logn)
  {
    IntegerRingRns res;
    return apply_op_inplace(
      res, [logn](auto x) { return decltype(x)::inv_log_size(logn); }, std::make_index_sequence<NofFields>{});
  }

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
        std::make_index_sequence<NofFields>{});
    } else {
      // reduce directly from input to output
      apply_op_inplace(
        *rns_out_casted,
        [&](auto x) { return decltype(x)::from((std::byte*)zq, RNS_CONFIG::limbs_count * sizeof(uint32_t)); },
        std::make_index_sequence<NofFields>{});
    }
  }

  // Conversion to/from direct representation
  template <typename Zq>
  static constexpr HOST_DEVICE IntegerRingRns from_direct(const Zq& zq)
  {
    IntegerRingRns res;
    convert_direct_to_rns(&zq.limbs_storage, &res.limbs_storage);
    return res;
  }

  IntegerRingRns() = default;

  template <typename Zq>
  IntegerRingRns(const Zq& zq)
  {
    convert_direct_to_rns(&zq.limbs_storage, &limbs_storage);
  }
};

template <class CONFIG>
struct std::hash<IntegerRingRns<CONFIG>> {
  std::size_t operator()(const IntegerRingRns<CONFIG>& key) const
  {
    std::size_t hash = 0;
    // boost hashing, see
    // https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/35991300#35991300
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