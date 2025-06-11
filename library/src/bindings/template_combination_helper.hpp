#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <functional>

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
