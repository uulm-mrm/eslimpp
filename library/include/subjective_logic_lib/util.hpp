#pragma once

#include <iostream>
#include <type_traits>

#ifdef __CUDA_ARCH__
#define CUDA_AVAIL __host__ __device__
#else
#define CUDA_AVAIL
#endif

namespace subjective_logic
{
template <std::size_t N>
concept is_binomial = N == 2;

/** @defgroup EPS_group Epsilon Definitions used within eSLIM++
 *  test group desription
 *  @{
 */

/**
 * @brief EPS allows to specifiy the epsilon used during comparisons within eSLIM++
 *        If no type specific value is specified, the std::nummeric_limits<T>::epsilon is used
 */
template <typename FloatT>
struct EPS
{
  static constexpr FloatT epsilon = std::numeric_limits<FloatT>::epsilon();
};

/**
 * @brief specific epsilon value for double type comparisons within eSLIM++
 */
template <>
struct EPS<double>
{
  static constexpr double value{ 1e-10 };
};
/**
 * @brief specific epsilon value for float type comparisons within eSLIM++
 */
template <>
struct EPS<float>
{
  static constexpr float value{ 1e-5 };
};

/**
 * @brief shortcut definition to access the value of a specific EPS struct
 */
template <typename FloatT>
static constexpr FloatT EPS_v{ EPS<FloatT>::value };
/** @} */

/** @defgroup OperatorConcepts concepts for operator definitions
 *  these concepts help checking for convertibility when declaring operators
 *  @{
 */
template <typename T, typename U>
concept is_addable = requires(T a, U b) { a + b; };
template <typename T, typename U>
concept is_substractable = requires(T a, U b) { a - b; };
template <typename T, typename U>
concept is_multipliable = requires(T a, U b) { a* b; };
template <typename T, typename U>
concept is_dividable = requires(T a, U b) { a / b; };
/** @} */

/** @defgroup HelperConcept concepts for function definitions
 *  the concepts below mostly check for different Opinion types or types within lists
 *  @{
 */
template <typename F, typename...>
struct FirstType
{
  using type = F;
};

template <std::size_t F, std::size_t...>
struct FirstNumber
{
  static constexpr std::size_t First = F;
};

template <typename Ref, typename... OpinionList>
concept is_list_of = (std::is_same_v<Ref, OpinionList> && ...);

template <typename... OpinionList>
concept is_list_of_same = (std::is_same_v<typename FirstType<OpinionList...>::type, OpinionList> && ...);

template <typename... OpinionList>
concept is_floating_point_list =
    is_list_of_same<OpinionList...> and std::is_floating_point_v<typename FirstType<OpinionList...>::type>;

template <typename... OpinionList>
concept is_arithmetic_list =
    is_list_of_same<OpinionList...> and std::is_arithmetic_v<typename FirstType<OpinionList...>::type>;

template <std::size_t N, typename FloatT>
class OpinionNoBase;
template <std::size_t N, typename FloatT>
class Opinion;

template <typename OpinionT>
class TrustedOpinion;

template <typename OpinionT>
concept is_opinion_no_base = std::is_same_v<OpinionT, OpinionNoBase<OpinionT::SIZE, typename OpinionT::FLOAT_t>>;
template <typename OpinionT>
concept is_opinion = std::is_same_v<OpinionT, Opinion<OpinionT::SIZE, typename OpinionT::FLOAT_t>>;
template <typename TrustedOp>
concept is_trusted_opinion =
    (is_opinion<typename TrustedOp::OpinionT> or is_opinion_no_base<typename TrustedOp::OpinionT>) and
    std::is_same_v<TrustedOp, TrustedOpinion<typename TrustedOp::OpinionT>>;

template <typename... OpinionList>
concept is_opinion_no_base_list =
    (is_opinion_no_base<typename FirstType<OpinionList...>::type> && is_list_of_same<OpinionList...>);
template <typename... OpinionList>
concept is_opinion_list = (is_opinion<typename FirstType<OpinionList...>::type> && is_list_of_same<OpinionList...>);
template <typename... OpinionList>
concept is_trusted_opinion_list =
    (is_trusted_opinion<typename FirstType<OpinionList...>::type> && is_list_of_same<OpinionList...>);
/** @} */

/** @defgroup ConstExprHelper constexpr definitions
 *  the definitions below allow for some constexpr functions like a for loop like call or min/max operations
 *  @{
 */
/**
 * @brief constexpr for loop-like call of a function, loop is executed in a templated recursive manner allowing
 * unrolling the loop
 * @tparam Start - start value of the for-loop
 * @tparam End - End value of the for-loop
 * @tparam Inc - increment used between different loop iterations
 * @tparam TYPES - list of types that are passed to the func in addition to the index, autogenerated in usual call
 * @param func - function which gets called with in increasing index
 * @param types - list of arbitrary parameters that are passed to func after the index of the current loop
 */
template <auto Start, auto End, auto Inc = 1, typename... TYPES, typename Func>
CUDA_AVAIL constexpr void constexpr_for(Func&& func, TYPES... types)
{
  if constexpr (Start < End)
  {
    func(Start, types...);

    // use explicit check before creating template function...
    // when not checking here, the functionality might be the same, however, a function is created with Start=End.
    // thus, even that it is within if constexpr, an invalid function call syntax is created but not executed.
    // unfortunately, by this, the compiler can detect invalid array accesses and produce warnings
    if constexpr (Start + Inc < End)
    {
      constexpr_for<Start + Inc, End, Inc, TYPES...>(func, types...);
    }
  }
}

/**
 * @brief searching the minimum of function return values, return types must be indirectly comparable
 * @tparam Start - start value of the for-loop
 * @tparam End - End value of the for-loop
 * @tparam Inc - increment used between different loop iterations
 * @tparam TYPES - list of types that are passed to the func in addition to the index, autogenerated in usual call
 * @param func - function which gets called with in increasing index and types. Return value used for comparison
 * @param types - list of arbitrary parameters that are passed to func after the index of the current loop
 */
template <auto Start, auto End, auto Inc = 1, typename... TYPES, typename Func>
CUDA_AVAIL auto min(Func&& func, TYPES... types)
{
  static_assert((End - Start) > 0);
  auto min = func(Start, types...);
  constexpr_for<Start + Inc, End, Inc>([&](std::size_t idx) {
    auto value = func(idx, types...);
    if (value < min)
    {
      min = value;
    }
  });
  return min;
}

/**
 * @brief searching the maximum of function return values, return types must be indirectly comparable
 * @tparam Start - start value of the for-loop
 * @tparam End - End value of the for-loop
 * @tparam Inc - increment used between different loop iterations
 * @tparam TYPES - list of types that are passed to the func in addition to the index, autogenerated in usual call
 * @param func - function which gets called with in increasing index and types. Return value used for comparison
 * @param types - list of arbitrary parameters that are passed to func after the index of the current loop
 */
template <auto Start, auto End, auto Inc = 1, typename... TYPES, typename Func>
CUDA_AVAIL auto max(Func&& func, TYPES... types)
{
  static_assert((End - Start) > 0);
  auto max = func(Start, types...);
  constexpr_for<Start + Inc, End, Inc>([&](std::size_t idx) {
    auto value = func(idx, types...);
    if (value > max)
    {
      max = value;
    }
  });
  return max;
}
/** @} */

}  // namespace subjective_logic