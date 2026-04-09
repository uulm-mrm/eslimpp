#include "container_bindings.hpp"

#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/container/long_short_term_memory.hpp"

#include <nanobind/stl/function.h>

namespace nb = nanobind;
namespace sl = subjective_logic;

template <std::size_t N, typename FloatT>
struct LSTMemoryLoader
{
  static void load(::nanobind::module_& bound_module)
  {
    using OpinionT = sl::Opinion<N, FloatT>;
    using LSTM = sl::container::LongShortTermMemory<OpinionT>;
    std::string module_name{ "LongShortTermMemory" };
    module_name += std::to_string(N);
    if constexpr (std::is_same_v<FloatT, double>)
    {
      module_name += "d";
    }
    else
    {
      module_name += "f";
    }

    auto bound_class = nb::class_<LSTM>(bound_module, module_name.c_str())
                           .def(nb::init<std::size_t, FloatT, FloatT, typename LSTM::FusionFunc>())
                           .def("add", &LSTM::add)
                           .def("is_last_conflicted", &LSTM::is_last_conflicted)
                           .def("get_opinion", &LSTM::get_opinion)
                           .def("get_long_opinion", &LSTM::get_long_opinion)
                           .def("get_short_opinion", &LSTM::get_short_opinion);
    //            .def(nb::init<Array>())
    //            .def(nb::init<Array, Array>())
    //            .def_static("from_evidences", &Dirichlet::from_evidences)
    //            .def_prop_rw(
    //                "evidences",
    //                [](Dirichlet& dir) -> WeightType& { return dir.evidences(); },
    //                [](Dirichlet dir, WeightType weights) { dir.evidences() = weights; },
    //                nb::rv_policy::reference)
    //            .def("evidences_copy", nb::overload_cast<>(&Dirichlet::evidences, nb::const_))
    //            .def_prop_rw(
    //                "priors",
    //                [](Dirichlet& dir) -> WeightType& { return dir.priors(); },
    //                [](Dirichlet dir, WeightType weights) { dir.priors() = weights; },
    //                nb::rv_policy::reference)
    //            .def("priors_copy", nb::overload_cast<>(&Dirichlet::priors, nb::const_))
    //            .def("alphas", &Dirichlet::alphas)
    //            .def("as_opinion", [](Dirichlet& dir) { return static_cast<sl::Opinion<N, FloatT>>(dir); })
    //            .def("as_opinion_no_base", [](Dirichlet& dir) { return static_cast<sl::OpinionNoBase<N, FloatT>>(dir);
    //            }) .def("evaluate", nb::overload_cast<WeightType>(&Dirichlet::evaluate, nb::const_)) .def("mean",
    //            &Dirichlet::mean) .def("variances", &Dirichlet::variance) .def("moment_matching_update_",
    //            &Dirichlet::moment_matching_update_, nb::rv_policy::reference) .def("moment_matching_update",
    //            &Dirichlet::moment_matching_update) .def("copy", [](const Dirichlet& dir) -> Dirichlet { return dir;
    //            });
  }
};

void loadLongShortTermMemoryBindings(::nanobind::module_& bound_module)
{
  loadBindings<LSTMemoryLoader>(bound_module);
}
