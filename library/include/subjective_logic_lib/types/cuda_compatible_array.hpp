#pragma once

#include <cstring>
#include <array>
#include <vector>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/types/cuda_compatible_iterator.hpp"

namespace subjective_logic
{

/**
 * using std::array might seem like the obvious option over creating a separate array again.
 * however, when using the subjective logic implementation with a std::array, cuda kernels started to behave weirdly.
 * thus, since we didn't want to introduce thrust to this library, we decided to replicate the most important properties
 * of the std::array.
 *
 * IMPORTANT!!!
 * implementing a ctor given an initializer list led to several issues...
 * first, the ctor below allowing an arbitrary number of arguments can no longer be called with {}
 * (even tho it's better in some cases (constexpr))
 * second, the std::initializer_list is not constexpr, so the ctor doesn't allow constexpr use with {}
 *
 * @tparam N - dimension of the array
 * @tparam T - entry_type
 */
template <std::size_t N, typename T>
struct Array
{
  template <std::size_t M, typename U>
  friend struct Array;

  using iterator = IteratorClass<T, false>;
  using reverse_iterator = IteratorClass<T, true>;
  using const_iterator = IteratorClass<const T, false>;
  using const_reverse_iterator = IteratorClass<const T, true>;

  /**
   * @brief constexpr access to the template parameter N, similar interface to std container
   */
  CUDA_AVAIL
  static constexpr std::size_t size()
  {
    return N;
  }
  using value_type = T;

  /**
   * @brief default ctor will fill every entry explicitly with 0, if T is constructable with 0
   */
  CUDA_AVAIL
  constexpr Array()
    requires requires { T{ 0 }; };

  /**
   * @brief default ctor will fill every entry using the default ctor of T
   */
  CUDA_AVAIL
  constexpr Array()
    requires(!requires { T{ 0 }; });

  /**
   * @brief default_entry is copy assigned to each element
   * @param default_entry - default value copied to each entry
   */
  CUDA_AVAIL
  constexpr explicit Array(T default_entry);

  /**
   * @brief ctor with implicit value conversion
   */
  template <typename OtherT>
  CUDA_AVAIL explicit constexpr Array(Array<N, OtherT> other)
    requires is_static_castable<OtherT, T>;

  constexpr Array(const Array& other) = default;
  constexpr Array(Array&& other) = default;

  /**
   * @brief ctor for compatibility reasons to convert std::arrays; not available in CUDA obviously
   * @param other - std::array that gets copied
   */
  explicit constexpr Array(std::array<T, N> other);

  /**
   * @brief ctor allowing a list of values without the use of {}
   */
  template <typename... VALUES>
  CUDA_AVAIL constexpr explicit Array(VALUES... values)
    requires(sizeof...(VALUES) == N and (is_static_castable<VALUES, T> && ...));

  /**
   * @brief factory function allowing the initialization with a value list
   */
  template <typename... VALUES>
  CUDA_AVAIL constexpr void array_factory(T value, VALUES... values)
    requires(sizeof...(VALUES) == 0 or (is_static_castable<VALUES, T> && ...));

  constexpr Array& operator=(const Array& other) = default;
  constexpr Array& operator=(Array&& other) = default;

  CUDA_AVAIL
  constexpr iterator begin();
  CUDA_AVAIL
  [[nodiscard]] constexpr const_iterator begin() const;
  CUDA_AVAIL
  constexpr iterator end();
  CUDA_AVAIL
  [[nodiscard]] constexpr const_iterator end() const;

  CUDA_AVAIL
  constexpr reverse_iterator rbegin();
  CUDA_AVAIL
  constexpr const_reverse_iterator rbegin() const;
  CUDA_AVAIL
  constexpr reverse_iterator rend();
  CUDA_AVAIL
  constexpr const_reverse_iterator rend() const;

  CUDA_AVAIL
  constexpr T& front();
  CUDA_AVAIL
  constexpr const T& front() const;
  CUDA_AVAIL
  constexpr T& back();
  CUDA_AVAIL
  constexpr const T& back() const;

  CUDA_AVAIL
  constexpr void fill(T entry);

  CUDA_AVAIL
  constexpr T& operator[](std::size_t idx);
  CUDA_AVAIL
  constexpr const T& operator[](std::size_t idx) const;

  explicit operator std::string() const;

  [[nodiscard]] std::string to_string() const;

  /**
   * @brief convert functions to stl container, not available with CUDA
   */
  std::vector<T> as_vector() const;
  /**
   * @brief convert functions to stl container, not available with CUDA
   */
  std::array<T, N> as_array() const;

  /**
   * @brief get direct const access to the entries to avoid copy to array for raw access
   */
  CUDA_AVAIL
  auto const& entries() const
  {
    return entries_;
  }

  ///////////////
  // all non assigning operators with scalar values are defined outside class
  // partly necessary, to allow left hand side multiplication
  ///////////////

  ///@{
  /** operators for applying math operations element wise */
  template <typename U>
  CUDA_AVAIL constexpr Array operator+(const Array<N, U>& other) const
    requires is_addable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator+=(const Array<N, U>& other)
    requires is_addable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator+=(const U& value)
    requires is_addable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array operator-(const Array<N, U>& other) const
    requires is_substractable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator-=(const Array<N, U>& other)
    requires is_substractable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator-=(const U& value)
    requires is_substractable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array operator*(const Array<N, U>& other) const
    requires is_multipliable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator*=(const Array<N, U>& other)
    requires is_multipliable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator*=(const U& value)
    requires is_multipliable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array operator/(const Array<N, U>& other) const
    requires is_dividable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator/=(const Array<N, U>& other)
    requires is_dividable<T, U>;

  template <typename U>
  CUDA_AVAIL constexpr Array& operator/=(const U& value)
    requires is_dividable<T, U>;
  ///@}

  template <typename OtherT>
  CUDA_AVAIL constexpr explicit operator Array<N, OtherT>() const
    requires is_static_castable<T, OtherT>;

  CUDA_AVAIL
  constexpr bool operator==(const Array& other) const;

  /**
   * @brief summing all elements in the array
   */
  CUDA_AVAIL
  constexpr T sum() const
    requires requires(T a, T b) { a + b; };

protected:
  T entries_[N];
};

template <std::size_t N, typename T>
constexpr Array<N, T>::Array()
  requires requires { T{ 0 }; }
{
  fill(T{ 0 });
}

template <std::size_t N, typename T>
constexpr Array<N, T>::Array()
  requires(!requires { T{ 0 }; })
{
  fill(T{});
}

template <std::size_t N, typename T>
constexpr Array<N, T>::Array(T default_entry)
{
  fill(default_entry);
}

template <std::size_t N, typename T>
template <typename OtherT>
constexpr Array<N, T>::Array(Array<N, OtherT> other)
  requires is_static_castable<OtherT, T>
{
  constexpr_for<0, N>([this, other](std::size_t idx) { entries_[idx] = static_cast<T>(other.entries_[idx]); });
}

template <std::size_t N, typename T>
template <typename OtherT>
constexpr Array<N, T>::operator Array<N, OtherT>() const
  requires is_static_castable<T, OtherT>
{
  return Array<N, OtherT>(*this);
}

template <std::size_t N, typename T>
constexpr Array<N, T>::Array(std::array<T, N> other)
{
  std::memcpy(entries_, other.data(), sizeof(T) * other.size());
}

template <std::size_t N, typename T>
template <typename... VALUES>
constexpr Array<N, T>::Array(VALUES... values)
  requires(sizeof...(VALUES) == N and (is_static_castable<VALUES, T> && ...))
{
  array_factory(values...);
}

template <std::size_t N, typename T>
template <typename... VALUES>
constexpr void Array<N, T>::array_factory(T value, VALUES... values)
  requires(sizeof...(VALUES) == 0 or (is_static_castable<VALUES, T> && ...))
{
  constexpr std::size_t idx{ sizeof...(VALUES) };

  entries_[(N - 1) - idx] = static_cast<T>(value);
  if constexpr (idx > 0)
  {
    array_factory(values...);
  }
}

template <std::size_t N, typename T>
constexpr void Array<N, T>::fill(T entry)
{
  constexpr_for<0, N, 1>([this, entry](std::size_t idx) { entries_[idx] = entry; });
}

template <std::size_t N, typename T>
constexpr Array<N, T>::iterator Array<N, T>::begin()
{
  return iterator{ &entries_[0] };
}

template <std::size_t N, typename T>
constexpr Array<N, T>::const_iterator Array<N, T>::begin() const
{
  return const_iterator{ &entries_[0] };
}

template <std::size_t N, typename T>
constexpr Array<N, T>::iterator Array<N, T>::end()
{
  return iterator{ &entries_[N - 1] + 1 };
}

template <std::size_t N, typename T>
constexpr Array<N, T>::const_iterator Array<N, T>::end() const
{
  return const_iterator{ &entries_[N - 1] + 1 };
}

template <std::size_t N, typename T>
constexpr Array<N, T>::reverse_iterator Array<N, T>::rbegin()
{
  return reverse_iterator{ &entries_[N - 1] };
}

template <std::size_t N, typename T>
constexpr Array<N, T>::const_reverse_iterator Array<N, T>::rbegin() const
{
  return const_reverse_iterator{ &entries_[N - 1] };
}

template <std::size_t N, typename T>
constexpr Array<N, T>::reverse_iterator Array<N, T>::rend()
{
  return reverse_iterator{ &entries_[0] - 1 };
}

template <std::size_t N, typename T>
constexpr Array<N, T>::const_reverse_iterator Array<N, T>::rend() const
{
  return const_reverse_iterator{ &entries_[0] - 1 };
}

template <std::size_t N, typename T>
constexpr T& Array<N, T>::front()
{
  return entries_[0];
}

template <std::size_t N, typename T>
constexpr const T& Array<N, T>::front() const
{
  return entries_[0];
}

template <std::size_t N, typename T>
constexpr T& Array<N, T>::back()
{
  return entries_[N - 1];
}

template <std::size_t N, typename T>
constexpr const T& Array<N, T>::back() const
{
  return entries_[N - 1];
}

template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator+(U value, Array<N, T> array)
  requires is_addable<U, T> and (not std::is_same_v<std::remove_reference_t<U>, Array<N, T>>)
{
  return array += value;
}
template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator+(Array<N, T> array, U value)
  requires is_addable<T, U> and (not std::is_same_v<std::remove_reference_t<U>, Array<N, T>>)
{
  return array += value;
}

template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator-(U value, Array<N, T> array)
  requires is_substractable<U, T>
{
  constexpr_for<0, N>([&] CUDA_AVAIL(std::size_t idx) { array[idx] = value - array[idx]; });
  return array;
  // return value + ((-1) * array); although possible, it would make use of multiple loops
}
template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator-(Array<N, T> array, U value)
  requires is_substractable<T, U>
{
  return array -= value;
}

template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator*(U value, Array<N, T> array)
  requires is_multipliable<U, T>
{
  return array *= value;
}
template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator*(Array<N, T> array, U value)
  requires is_multipliable<T, U>
{
  return array *= value;
}

template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator/(U value, Array<N, T> array)
  requires is_dividable<U, T>
{
  constexpr_for<0, N>([&] CUDA_AVAIL(std::size_t idx) { array[idx] = value / array[idx]; });
  return array;
}
template <std::size_t N, typename T, typename U>
CUDA_AVAIL Array<N, T> operator/(Array<N, T> array, U value)
  requires is_dividable<T, U>
{
  return array /= value;
}

template <std::size_t N, typename T>
template <typename U>
CUDA_AVAIL constexpr Array<N, T> Array<N, T>::operator+(const Array<N, U>& other) const
  requires is_addable<T, U>
{
  return Array(*this).operator+=(other);
}
template <std::size_t N, typename T>
template <typename U>
CUDA_AVAIL constexpr Array<N, T>& Array<N, T>::operator+=(const Array<N, U>& other)
  requires is_addable<T, U>
{
  constexpr_for<0, N>([this, &other](std::size_t idx) { entries_[idx] += other[idx]; });
  return *this;
}
template <std::size_t N, typename T>
template <typename U>
CUDA_AVAIL constexpr Array<N, T>& Array<N, T>::operator+=(const U& value)
  requires is_addable<T, U>
{
  constexpr_for<0, N>([this, value](std::size_t idx) { entries_[idx] += value; });
  return *this;
}

template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T> Array<N, T>::operator-(const Array<N, U>& other) const
  requires is_substractable<T, U>
{
  return Array(*this).operator-=(other);
}
template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T>& Array<N, T>::operator-=(const Array<N, U>& other)
  requires is_substractable<T, U>
{
  constexpr_for<0, N>([this, &other](std::size_t idx) { entries_[idx] -= other[idx]; });
  return *this;
}
template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T>& Array<N, T>::operator-=(const U& value)
  requires is_substractable<T, U>
{
  constexpr_for<0, N>([this, value](std::size_t idx) { entries_[idx] -= value; });
  return *this;
}

template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T> Array<N, T>::operator*(const Array<N, U>& other) const
  requires is_multipliable<T, U>
{
  return Array(*this).operator*=(other);
}
template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T>& Array<N, T>::operator*=(const Array<N, U>& other)
  requires is_multipliable<T, U>
{
  constexpr_for<0, N>([this, &other](std::size_t idx) { entries_[idx] *= other[idx]; });
  return *this;
}
template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T>& Array<N, T>::operator*=(const U& value)
  requires is_multipliable<T, U>
{
  constexpr_for<0, N>([this, value](std::size_t idx) { entries_[idx] *= value; });
  return *this;
}

template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T> Array<N, T>::operator/(const Array<N, U>& other) const
  requires is_dividable<T, U>
{
  return Array(*this).operator/=(other);
}
template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T>& Array<N, T>::operator/=(const Array<N, U>& other)
  requires is_dividable<T, U>
{
  constexpr_for<0, N>([this, &other](std::size_t idx) { entries_[idx] /= other[idx]; });
  return *this;
}
template <std::size_t N, typename T>
template <typename U>
constexpr Array<N, T>& Array<N, T>::operator/=(const U& value)
  requires is_dividable<T, U>
{
  constexpr_for<0, N>([this, value](std::size_t idx) { entries_[idx] /= value; });
  return *this;
}

template <std::size_t N, typename T>
constexpr T Array<N, T>::sum() const
  requires requires(T a, T b) { a + b; }
{
#ifdef __CUDA_ARCH__
  T sum{ entries_[0] };
  for (int i{ 1 }; i < N; ++i)
  {
    sum += entries_[i];
  }
  return sum;
#else
  T sum{ entries_[0] };
  // start by 1 is indented, since sum already contains first element
  constexpr_for<1, N>([this, &sum](std::size_t idx) { sum += entries_[idx]; });
  return sum;
#endif
}

template <std::size_t N, typename T>
constexpr T& Array<N, T>::operator[](std::size_t idx)
{
  return entries_[idx];
}

template <std::size_t N, typename T>
constexpr const T& Array<N, T>::operator[](std::size_t idx) const
{
  return entries_[idx];
}

template <std::size_t N, typename T>
constexpr bool Array<N, T>::operator==(const Array& other) const
{
  bool equal{ true };
  constexpr_for<0, N>([this, &other, &equal](std::size_t idx) {
    using std::abs;
    equal &= abs(entries_[idx] - other.entries_[idx]) < EPS_v<T>;
  });
  return equal;
}

template <std::size_t N, typename T>
inline std::ostream& operator<<(std::ostream& out, Array<N, T> const& array)
{
  out << static_cast<std::string>(array);
  return out;
}

template <std::size_t N, typename T>
Array<N, T>::operator std::string() const
{
  return to_string();
}

template <std::size_t N, typename T>
std::string Array<N, T>::to_string() const
{
  std::string out{ "[" };
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    if (N > 0)
    {
      out += " ";
    }
    out += std::to_string(entries_[idx]);
  }
  out += "]";
  return out;
}

template <std::size_t N, typename T>
std::vector<T> Array<N, T>::as_vector() const
{
  std::vector<T> out;
  out.reserve(N);
  // if compiled in nvcc, the constexpr_for is also available on __device__ and therefor has issues with host functions
  // even on host... thus, a workaround is used here.
  constexpr_for_host<0, N>([this, &out](std::size_t idx) { out.push_back(this->entries_[idx]); });
  return out;
}

template <std::size_t N, typename T>
std::array<T, N> Array<N, T>::as_array() const
{
  std::array<T, N> out;
  constexpr_for_host<0, N>([this, &out](std::size_t idx) { out[idx] = this->entries_[idx]; });
  return out;
}

}  // namespace subjective_logic
