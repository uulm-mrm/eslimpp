#pragma once
#include "template_combination_helper.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <functional>
#include <tuple>

template <std::size_t... NUMBERS>
struct NumberList
{
};
template <typename... TYPES>
struct TypeList
{
};

template <template <std::size_t, typename> typename loader,
          typename T,
          std::size_t N,
          typename FloatT,
          typename... FloatTs>
constexpr void loadTypes(T& nb_mod)
{
  loader<N, FloatT>::load(nb_mod);
  if constexpr (sizeof...(FloatTs) > 0)
  {
    loadTypes<loader, T, N, FloatTs...>(nb_mod);
  }
}

template <template <std::size_t, typename> typename loader,
          typename T,
          std::size_t number,
          std::size_t... numbers,
          typename... TYPES,
          template <typename...>
          typename List>
constexpr void loadNumbers(T& nb_mod, const List<TYPES...>& types)
{
  loadTypes<loader, T, number, TYPES...>(nb_mod);
  if constexpr (sizeof...(numbers) > 0)
  {
    loadNumbers<loader, T, numbers...>(nb_mod, types);
  }
}

template <template <std::size_t, typename> typename loader, typename T, std::size_t... NUMBERS, typename... TYPES>
constexpr void loadCombination(T& nb_mod, const NumberList<NUMBERS...>& number, const TypeList<TYPES...>& types)
{
  loadNumbers<loader, T, NUMBERS...>(nb_mod, types);
}

template <template <std::size_t, typename> typename loader, typename T>
void loadBindings(T& nb_mod)
{
  NumberList<2, 3, 4, 5, 6, 7, 8, 9, 10> numbers;
  TypeList<float, double> types;
  loadCombination<loader>(nb_mod, numbers, types);
}

template <typename T, std::size_t>
using RepeatType = T;

template <typename T, std::size_t... INDICES>
auto make_tuple_with_type_and_length(std::index_sequence<INDICES...>)
    -> std::tuple<RepeatType<T, INDICES>...>;

template <typename T, std::size_t N>
using TupleWithTypeAnLength = decltype(make_tuple_with_type_and_length<T>(std::make_index_sequence<N>{}));

template <std::size_t N, typename FloatT, std::size_t... INDICES>
auto tuplefy_array(FloatT (&data)[N], std::index_sequence<INDICES...>)
{
  return std::make_tuple(data[INDICES]...);
}

template <std::size_t N, typename FloatT>
auto tuplefy_array(FloatT (&data)[N])
{
  return tuplefy_array(data, std::make_index_sequence<N>{});
}

template <std::size_t N, typename FloatT, std::size_t... INDICES>
auto tuplefy_array(const std::array<FloatT, N>& data, std::index_sequence<INDICES...>)
{
  return std::make_tuple(data[INDICES]...);
}
template <std::size_t N, typename FloatT>
auto tuplefy_array(const std::array<FloatT, N>& data)
{
  return tuplefy_array(data, std::make_index_sequence<N>{});
}

template <typename Tuple, std::size_t... INDICES>
auto arrayfy_tuple(Tuple&& data, std::index_sequence<INDICES...>)
{
  using EntryType = std::remove_cvref_t<std::tuple_element_t<0, std::remove_cvref_t<Tuple>>>;
  return std::array<EntryType, sizeof...(INDICES)>{ std::get<INDICES>(data)... };
}

template <typename Tuple>
auto arrayfy_tuple(Tuple&& data)
{
  return arrayfy_tuple(std::forward<Tuple>(data),
                       std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple>>>{});
}
