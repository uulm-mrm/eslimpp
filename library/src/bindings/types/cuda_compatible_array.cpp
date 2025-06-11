#include "types_bindings.hpp"

#include <string>

#include "subjective_logic_lib/types/cuda_compatible_array.hpp"
#include "nanobind/make_iterator.h"

namespace nb = nanobind;
namespace sl = subjective_logic;

template <std::size_t N, typename FloatT>
struct ArrayLoader
{
  template <typename... ARGS>
  static void defineArrayCtorArgs(::nanobind::class_<sl::Array<N, FloatT>>& bound_class)
  {
    if constexpr (sizeof...(ARGS) == N)
    {
      bound_class.def(nb::init<ARGS...>());
    }
    else
    {
      // add another FloatT to the list
      defineArrayCtorArgs<FloatT, ARGS...>(bound_class);
    }
  }

  static void load(::nanobind::module_& bound_module)
  {
    using Array = sl::Array<N, FloatT>;
    std::string module_name{ "Array" };
    module_name += std::to_string(N);
    if constexpr (std::is_same_v<FloatT, double>)
    {
      module_name += "d";
    }
    else
    {
      module_name += "f";
    }
    auto bound_class =
        nb::class_<Array>(bound_module, module_name.c_str())
            .def(nb::init_implicit<std::array<FloatT, N>>())
            .def(nb::init())
            .def(nb::init<const Array&>())
            .def(nb::init<Array&&>())
            .def(nb::init<std::array<FloatT, N>>())
            .def(nb::self + nb::self)
            .def(nb::self + FloatT())
            .def(nb::self += nb::self)
            .def(nb::self += FloatT())
            .def(nb::self - nb::self)
            .def(nb::self - FloatT())
            .def(nb::self -= nb::self)
            .def(nb::self -= FloatT())
            .def(nb::self * nb::self)
            .def(float() * nb::self)
            .def(FloatT() * nb::self)
            .def(nb::self * FloatT())
            .def(nb::self *= nb::self)
            .def(nb::self *= FloatT())
            .def(nb::self / nb::self)
            .def(nb::self / FloatT())
            .def(nb::self /= nb::self)
            .def(nb::self /= FloatT())
            .def("copy", [](const Array& a) -> Array { return a; })
            .def("__len__", [](const Array& a) { return a.size(); })
            .def("__repr__", &Array::to_string)
            .def("__deepcopy__", [](const Array& a, nb::dict memo) -> Array { return a; })
            .def(
                "__iter__",
                [](const Array& a) { return nb::make_iterator(nb::type<Array>(), "iterator", a.begin(), a.end()); },
                nb::keep_alive<0, 1>())
            .def(
                "__getitem__",
                [](const Array& a, int idx) {
                  if (idx < 0)
                  {
                    idx += a.size();
                  }
                  return a[idx];
                },
                nb::rv_policy::reference)
            .def("__setitem__", [](Array& a, int idx, FloatT value) {
              if (idx < 0)
              {
                idx += a.size();
              }
              a[idx] = value;
            });
    defineArrayCtorArgs(bound_class);
  }
};

void loadCudaCompatibleArrayBindings(::nanobind::module_& bound_module)
{
  loadBindings<ArrayLoader>(bound_module);
}
