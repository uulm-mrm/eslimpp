#pragma once

#include <cstring>

#include "subjective_logic_lib/util.hpp"

namespace subjective_logic
{

/**
 * @brief std iterator class used for the cuda_compatible_array defined explicitly to be compiled on CUDA and CXX
 */
template <typename T, bool Reverse = false>
struct IteratorClass
{
  using Reference = T&;
  using ConstReference = const T&;
  using Pointer = T*;
  using ConstPointer = const T*;

  CUDA_AVAIL
  explicit constexpr IteratorClass(Pointer pointer) : pointer_{ pointer }
  {
  }
  IteratorClass(const IteratorClass& other) = default;
  IteratorClass(IteratorClass&& other) = default;

  IteratorClass& operator=(const IteratorClass& other) = default;
  IteratorClass& operator=(IteratorClass&& other) = default;

  CUDA_AVAIL
  ConstReference operator*() const
  {
    return *pointer_;
  }
  CUDA_AVAIL
  Reference operator*()
  {
    return *pointer_;
  }
  CUDA_AVAIL
  ConstPointer operator->() const
  {
    return pointer_;
  }
  CUDA_AVAIL
  Pointer operator->()
  {
    return pointer_;
  }

  CUDA_AVAIL
  friend bool operator==(const IteratorClass& it_a, const IteratorClass& it_b)
  {
    return it_a.pointer_ == it_b.pointer_;
  };
  CUDA_AVAIL
  friend bool operator!=(const IteratorClass& it_a, const IteratorClass& it_b)
  {
    return it_a.pointer_ != it_b.pointer_;
  };

  CUDA_AVAIL
  IteratorClass& operator++()
  {
    if constexpr (Reverse)
    {
      --pointer_;
    }
    else
    {
      ++pointer_;
    }
    return *this;
  }
  CUDA_AVAIL
  IteratorClass operator++(int)
  {
    if constexpr (Reverse)
    {
      return IteratorClass{ pointer_-- };
    }
    else
    {
      return IteratorClass{ pointer_++ };
    }
  }

  CUDA_AVAIL
  IteratorClass& operator--()
  {
    if constexpr (Reverse)
    {
      ++pointer_;
    }
    else
    {
      --pointer_;
    }
    return *this;
  }

  CUDA_AVAIL
  IteratorClass operator--(int)
  {
    if constexpr (Reverse)
    {
      return IteratorClass{ pointer_++ };
    }
    return IteratorClass{ pointer_-- };
  }

  CUDA_AVAIL std::ptrdiff_t operator-(IteratorClass other)
  {
    if constexpr (Reverse)
    {
      return other.pointer_ - pointer_;
    }
    return pointer_ - other.pointer_;
  }

private:
  Pointer pointer_;
};

template <typename T, bool Reverse>
constexpr std::ptrdiff_t distance(IteratorClass<T, Reverse> it1, IteratorClass<T, Reverse> it2)
{
  return it2 - it1;
}

}  // namespace subjective_logic
