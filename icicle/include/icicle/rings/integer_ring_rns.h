#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include <array>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <utility>

// This class is a template class that implement integer-rings in RNS representation
template <typename RNS_CONFIG>
class IntegerRingRns
{
public:
  static constexpr unsigned limbs_count = RNS_CONFIG::limbs_count;
  using Fields = typename RNS_CONFIG::Fields;
  static constexpr unsigned NofFields = std::tuple_size_v<Fields>;
  static constexpr std::array<unsigned, limbs_count> FieldOffset = RNS_CONFIG::FieldOffset;
  template <size_t I>
  using FieldType = std::tuple_element_t<I, Fields>;

  storage<RNS_CONFIG::limbs_count> limbs_storage;

  // `get_field` with correct access to `limbs_storage.limbs`
  template <size_t I>
  HOST_DEVICE FieldType<I>* get_field()
  {
    return reinterpret_cast<FieldType<I>*>(limbs_storage.limbs + FieldOffset[I]);
  }

  template <size_t I>
  HOST_DEVICE const FieldType<I>* get_field() const
  {
    return reinterpret_cast<const FieldType<I>*>(limbs_storage.limbs + FieldOffset[I]);
  }

  // Zero Initialization
  template <size_t... I>
  static constexpr HOST_DEVICE IntegerRingRns zero_impl(std::index_sequence<I...>)
  {
    IntegerRingRns result;
    ((*(result.template get_field<I>()) = FieldType<I>::zero()), ...);
    return result;
  }

  static constexpr HOST_DEVICE IntegerRingRns zero() { return zero_impl(std::make_index_sequence<NofFields>{}); }

  // One Initialization
  template <size_t... I>
  static constexpr HOST_DEVICE IntegerRingRns one_impl(std::index_sequence<I...>)
  {
    IntegerRingRns result;
    ((*(result.template get_field<I>()) = FieldType<I>::one()), ...);
    return result;
  }

  static constexpr HOST_DEVICE IntegerRingRns one() { return one_impl(std::make_index_sequence<NofFields>{}); }

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

  // Generalized Arithmetic Implementation for `+`, `-`, `*`
  template <typename Op, size_t... I>
  static HOST_DEVICE IntegerRingRns
  apply_binary_op(const IntegerRingRns& a, const IntegerRingRns& b, Op op, std::index_sequence<I...>)
  {
    IntegerRingRns result;
    ((*(result.template get_field<I>()) = op(*(a.template get_field<I>()), *(b.template get_field<I>()))), ...);
    return result;
  }

  template <typename Op, size_t... I>
  static HOST_DEVICE IntegerRingRns apply_op_unary(const IntegerRingRns& x, Op op, std::index_sequence<I...>)
  {
    IntegerRingRns result;
    ((*(result.template get_field<I>()) = op(*(x.template get_field<I>()))), ...);
    return result;
  }

  template <typename Op, size_t... I>
  static HOST_DEVICE IntegerRingRns apply_op_inplace(IntegerRingRns& x, Op op, std::index_sequence<I...>)
  {
    ((*(x.template get_field<I>()) = op(*(x.template get_field<I>()))), ...);
    return x;
  }

  // Operator Overloads
  friend HOST_DEVICE IntegerRingRns operator+(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x + y; }, std::make_index_sequence<NofFields>{});
  }

  friend HOST_DEVICE IntegerRingRns operator-(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x - y; }, std::make_index_sequence<NofFields>{});
  }

  friend HOST_DEVICE IntegerRingRns operator*(const IntegerRingRns& a, const IntegerRingRns& b)
  {
    return apply_binary_op(a, b, [](auto x, auto y) { return x * y; }, std::make_index_sequence<NofFields>{});
  }

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
};