#pragma once

#include <cstdint>
#include <string>
#include <subjective_logic_lib/util.hpp>

namespace subjective_logic
{

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
struct LimitedFloat;

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr LimitT abs(LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> x);
template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr std::string to_string(LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> x);

template <typename T>
using ZeroOneFloat = LimitedFloat<T, static_cast<T>(0.0), static_cast<T>(1.0)>;

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
struct LimitedFloat
{
  using QuantizedT = std::uint8_t;
  using LIMIT_TYPE = LimitT;

  constexpr static LimitT LIMIT_LOWER_BOUND{ LOWER_BOUND };
  constexpr static LimitT LIMIT_UPPER_BOUND{ UPPER_BOUND };
  constexpr static LimitT LIMIT_RANGE{ UPPER_BOUND - LOWER_BOUND };

  constexpr static LimitT VALUE_RANGE{ std::numeric_limits<QuantizedT>::max() };

  constexpr LimitedFloat() = default;

  // ReSharper disable once CppNonExplicitConvertingConstructor

  // implicit assignment from LimitT is intended, which allows, e.g.,
  // LimitedFloat<double, 0.,1.0> lf_test = 0.5;
  // otherwise the float value must be cast explicitly every time.
  template <typename T>
  CUDA_AVAIL explicit constexpr LimitedFloat(T float_value)
    requires is_static_castable<T, LimitT>;

  template <typename T>
  CUDA_AVAIL explicit constexpr operator T() const
    requires is_static_castable<LimitT, T>;
  CUDA_AVAIL constexpr LimitT as_limit_type() const;

  template <typename T>
  CUDA_AVAIL constexpr LimitedFloat& operator=(T float_value)
    requires is_static_castable<T, LimitT>;

  template <typename T>
  CUDA_AVAIL constexpr LimitedFloat& operator+=(T other_value)
    requires is_addable<LimitT, T>;
  template <typename T>
  CUDA_AVAIL constexpr LimitT operator+(T other_value) const
    requires is_addable<LimitT, T>;

  CUDA_AVAIL constexpr LimitT operator+(LimitedFloat other_value) const;

  template <typename T>
  CUDA_AVAIL constexpr LimitedFloat& operator-=(T other_value)
    requires is_substractable<LimitT, T>;
  template <typename T>
  CUDA_AVAIL constexpr LimitT operator-(T other_value) const
    requires is_substractable<LimitT, T>;

  CUDA_AVAIL constexpr LimitT operator-(LimitedFloat other_value) const;

  template <typename T>
  CUDA_AVAIL constexpr LimitedFloat& operator*=(T other_value)
    requires is_multipliable<LimitT, T>;
  template <typename T>
  CUDA_AVAIL constexpr LimitT operator*(T other_value) const
    requires is_multipliable<LimitT, T>;

  CUDA_AVAIL constexpr LimitT operator*(LimitedFloat other_value) const;

  template <typename T>
  CUDA_AVAIL constexpr LimitedFloat& operator/=(T other_value)
    requires is_dividable<LimitT, T>;
  template <typename T>
  CUDA_AVAIL constexpr LimitT operator/(T other_value) const
    requires is_dividable<LimitT, T>;

  CUDA_AVAIL constexpr LimitT operator/(LimitedFloat other_value) const;

  CUDA_AVAIL constexpr auto operator<=>(LimitedFloat other_value) const;

  CUDA_AVAIL constexpr auto operator<=>(LimitT other_value) const;

  QuantizedT value;
};

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr LimitT abs(LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> x)
{
  return std::abs(x.as_limit_type());
}
template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr std::string to_string(LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> x)
{
  using std::to_string;
  return std::string("lFloat: ") + to_string(x.as_limit_type()) + " / " + to_string(x.LIMIT_LOWER_BOUND) + " < " +
         to_string(x.value) + " > " + to_string(x.LIMIT_UPPER_BOUND);
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::LimitedFloat(T float_value)
  requires is_static_castable<T, LimitT>
{
  // std::clamp not available with nvcc for cuda building on devices
  if (float_value < LOWER_BOUND)
  {
    float_value = LOWER_BOUND;
  }
  else if (float_value > UPPER_BOUND)
  {
    float_value = UPPER_BOUND;
  }
  value = (float_value - LOWER_BOUND) / LIMIT_RANGE * VALUE_RANGE;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator T() const
  requires is_static_castable<LimitT, T>
{
  return static_cast<T>(static_cast<LimitT>(value) / VALUE_RANGE * LIMIT_RANGE + LOWER_BOUND);
}
template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::as_limit_type() const
{
  return static_cast<LimitT>(*this);
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>&
LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator=(T float_value)
  requires is_static_castable<T, LimitT>
{
  *this = LimitedFloat(static_cast<LimitT>(float_value));
  return *this;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND, typename T>
constexpr LimitT operator+(T value, LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> lf_value)
  requires is_addable<T, LimitT>
{
  return value + lf_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>&
LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator+=(T other_value)
  requires is_addable<LimitT, T>
{
  *this = LimitedFloat{ *this + other_value };
  return *this;
}
template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator+(T other_value) const
  requires is_addable<LimitT, T>
{
  return this->as_limit_type() + other_value;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator+(LimitedFloat other_value) const
{
  return this->as_limit_type() + other_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND, typename T>
constexpr LimitT operator-(T value, LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> lf_value)
  requires is_substractable<T, LimitT>
{
  return value - lf_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>&
LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator-=(T other_value)
  requires is_substractable<LimitT, T>
{
  *this = LimitedFloat{ *this - other_value };
  return *this;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator-(T other_value) const
  requires is_substractable<LimitT, T>
{
  return this->as_limit_type() - other_value;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator-(LimitedFloat other_value) const
{
  return this->as_limit_type() - other_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND, typename T>
constexpr LimitT operator*(T value, LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> lf_value)
  requires is_multipliable<T, LimitT>
{
  return value * lf_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>&
LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator*=(T other_value)
  requires is_multipliable<LimitT, T>
{
  *this = LimitedFloat{ *this * other_value };
  return *this;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator*(T other_value) const
  requires is_multipliable<LimitT, T>
{
  return this->as_limit_type() * other_value;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator*(LimitedFloat other_value) const
{
  return this->as_limit_type() * other_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND, typename T>
constexpr LimitT operator/(T value, LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND> lf_value)
  requires is_dividable<T, LimitT>
{
  return value / lf_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>&
LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator/=(T other_value)
  requires is_dividable<LimitT, T>
{
  *this = LimitedFloat{ *this / other_value };
  return *this;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
template <typename T>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator/(T other_value) const
  requires is_dividable<LimitT, T>
{
  return this->as_limit_type() / other_value;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr LimitT LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator/(LimitedFloat other_value) const
{
  return this->as_limit_type() / other_value.as_limit_type();
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr auto LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator<=>(LimitedFloat other_value) const
{
  if (value < other_value.value)
  {
    return -1;
  }
  if (value > other_value.value)
  {
    return 1;
  }
  return 0;
}

template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
constexpr auto LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>::operator<=>(LimitT other_value) const
{
  if (static_cast<LimitT>(*this) < other_value)
  {
    return -1;
  }
  if (static_cast<LimitT>(*this) > other_value)
  {
    return 1;
  }
  return 0;
}

}  // namespace subjective_logic

// specialize numeric_limits for LimitedFloat
namespace std
{
template <class LimitT, LimitT LOWER_BOUND, LimitT UPPER_BOUND>
struct numeric_limits<subjective_logic::LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>>
{
  using lFloatT = subjective_logic::LimitedFloat<LimitT, LOWER_BOUND, UPPER_BOUND>;

  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = (UPPER_BOUND == std::numeric_limits<LimitT>::infinity());
  static constexpr int radix = 2;
  static constexpr int digits = 0;
  static constexpr int digits10 = 0;

  static constexpr LimitT min() noexcept
  {
    return LOWER_BOUND;
  }
  static constexpr LimitT max() noexcept
  {
    return LOWER_BOUND;
  }
  static constexpr LimitT lowest() noexcept
  {
    return min();
  }
  static constexpr LimitT epsilon() noexcept
  {
    return 1.0 / std::numeric_limits<typename lFloatT::QuantizedT>::max();
  }
};
}  // namespace std