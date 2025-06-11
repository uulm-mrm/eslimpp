#include "opinions_bindings.hpp"

#include <string>

#include "subjective_logic_lib/opinions/trusted_opinion.hpp"

namespace nb = nanobind;
namespace sl = subjective_logic;

template <std::size_t N, typename FloatT>
struct TrustedOpinionLoader
{
  template <typename OpinionT>
  static void loadClass(::nanobind::module_& bound_module)
  {
    std::string module_name{ "TrustedOpinion" };
    if constexpr (sl::is_opinion_no_base<OpinionT>)
    {
      module_name += "NoBase";
    }
    module_name += std::to_string(N);
    if constexpr (std::is_same_v<FloatT, double>)
    {
      module_name += "d";
    }
    else
    {
      module_name += "f";
    }

    using TOp = sl::TrustedOpinion<OpinionT>;
    using TrustT = typename TOp::TrustT;

    auto bound_class =
        nb::class_<TOp>(bound_module, module_name.c_str())
            .def_ro_static("dimension", &TOp::SIZE)
            .def_static("extractOpinions", &TOp::extractOpinions)
            .def_static("extractTrusts", &TOp::extractTrusts)
            .def_static("extractDiscountedOpinions", &TOp::extractDiscountedOpinions)
            .def(nb::init())
            .def(nb::init<TrustT, OpinionT>(), nb::arg("belief_masses"), nb::arg("prior_belief_masses"))
            .def("__deepcopy__", [](const TOp& a, nb::dict memo) -> TOp { return a; })
            // access to prop with reference is important!!
            // otherwise, t_opinion.trust.trust_discount_(0.9) inside python would update trust of t_opinion
            .def_prop_rw(
                "opinion",
                [](TOp& t_opin) -> TOp::OpinionT& { return t_opin.opinion(); },
                [](TOp& t_opin, TOp::OpinionT& opin) { t_opin.opinion() = opin; },
                nb::rv_policy::reference)
            .def("opinion_copy", nb::overload_cast<>(&TOp::opinion, nb::const_))
            .def_prop_rw(
                "trust",
                [](TOp& t_opin) -> TOp::TrustT& { return t_opin.trust(); },
                [](TOp& t_opin, TOp::TrustT& trust) { t_opin.trust() = trust; },
                nb::rv_policy::reference)
            .def("trust_copy", nb::overload_cast<>(&TOp::trust, nb::const_))
            .def("discounted_opinion", &TOp::discounted_opinion)
            .def("revise_trust_", nb::overload_cast<FloatT>(&TOp::revise_trust_), nb::rv_policy::reference)
            .def("revise_trust", nb::overload_cast<FloatT>(&TOp::revise_trust, nb::const_))
            .def("revise_trust_", nb::overload_cast<TOp&>(&TOp::revise_trust_), nb::rv_policy::reference)
            .def("revise_trust", nb::overload_cast<TOp>(&TOp::revise_trust, nb::const_))
            .def("__repr__", &TOp::to_string)
            .def("copy", [](const TOp& top) -> TOp { return top; });
  }

  static void load(::nanobind::module_& bound_module)
  {
    using OpinionT = sl::Opinion<N, FloatT>;
    using OpinionNoBaseT = sl::OpinionNoBase<N, FloatT>;

    loadClass<OpinionT>(bound_module);
    loadClass<OpinionNoBaseT>(bound_module);
  }
};

void loadTrustedOpinionBindings(::nanobind::module_& bound_module)
{
  loadBindings<TrustedOpinionLoader>(bound_module);
}
