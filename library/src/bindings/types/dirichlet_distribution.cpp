#include "types_bindings.hpp"

#include <string>

#include "subjective_logic_lib/types/dirichlet_distribution.hpp"
#include "nanobind/make_iterator.h"

namespace nb = nanobind;
namespace sl = subjective_logic;

template <std::size_t N, typename FloatT>
struct DirichletLoader
{
  template <typename... ARGS>
  static void defineArrayCtorArgs(::nanobind::class_<sl::DirichletDistribution<N, FloatT>>& bound_class)
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

  static void defineBinomialDependentFields(::nanobind::class_<sl::DirichletDistribution<N, FloatT>>& bound_class)
    requires sl::is_binomial<N>
  {
    using Dirichlet = sl::DirichletDistribution<N, FloatT>;
    // using WeightType = typename Dirichlet::WeightType;
    // using Array = sl::Array<N, FloatT>;
    bound_class.def("evaluate", nb::overload_cast<FloatT>(&Dirichlet::evaluate, nb::const_))
        .def("mean_binomial", &Dirichlet::mean_binomial);
  }

  static void defineBinomialDependentFields(::nanobind::class_<sl::DirichletDistribution<N, FloatT>>& bound_class)
    requires(not sl::is_binomial<N>)
  {
    //  using Opinion = sl::OpinionNoBase<N, FloatT>;
    //  ;
  }

  static void load(::nanobind::module_& bound_module)
  {
    using Dirichlet = sl::DirichletDistribution<N, FloatT>;
    using WeightType = typename Dirichlet::WeightType;
    using Array = sl::Array<N, FloatT>;
    std::string module_name{ "DirichletDistribution" };
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
        nb::class_<Dirichlet>(bound_module, module_name.c_str())
            .def(nb::init())
            .def(nb::init<Array>())
            .def(nb::init<Array, Array>())
            .def_static("from_evidences", &Dirichlet::from_evidences)
            .def_prop_rw(
                "evidences",
                [](Dirichlet& dir) -> WeightType& { return dir.evidences(); },
                [](Dirichlet dir, WeightType weights) { dir.evidences() = weights; },
                nb::rv_policy::reference)
            .def("evidences_copy", nb::overload_cast<>(&Dirichlet::evidences, nb::const_))
            .def_prop_rw(
                "priors",
                [](Dirichlet& dir) -> WeightType& { return dir.priors(); },
                [](Dirichlet dir, WeightType weights) { dir.priors() = weights; },
                nb::rv_policy::reference)
            .def("priors_copy", nb::overload_cast<>(&Dirichlet::priors, nb::const_))
            .def("alphas", &Dirichlet::alphas)
            .def("as_opinion", [](Dirichlet& dir) { return static_cast<sl::Opinion<N, FloatT>>(dir); })
            .def("as_opinion_no_base", [](Dirichlet& dir) { return static_cast<sl::OpinionNoBase<N, FloatT>>(dir); })
            .def("evaluate", nb::overload_cast<WeightType>(&Dirichlet::evaluate, nb::const_))
            .def("mean", &Dirichlet::mean)
            .def("variances", &Dirichlet::variance)
            .def("moment_matching_update_", &Dirichlet::moment_matching_update_, nb::rv_policy::reference)
            .def("moment_matching_update", &Dirichlet::moment_matching_update)
            .def("copy", [](const Dirichlet& dir) -> Dirichlet { return dir; });
    defineArrayCtorArgs(bound_class);
    defineBinomialDependentFields(bound_class);
  }
};

void loadDirichletDistributionBindings(::nanobind::module_& bound_module)
{
  loadBindings<DirichletLoader>(bound_module);
}
