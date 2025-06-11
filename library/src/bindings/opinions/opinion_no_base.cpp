#include "opinions_bindings.hpp"

#include <string>

#include "subjective_logic_lib/opinions/opinion_no_base.hpp"
#include "subjective_logic_lib/types/dirichlet_distribution.hpp"

namespace nb = nanobind;
namespace sl = subjective_logic;

template <std::size_t N, typename FloatT>
struct OpinionNoBaseLoader
{
  template <typename... ARGS>
  static void defineOpinionNoBaseCtorArgs(::nanobind::class_<sl::OpinionNoBase<N, FloatT>>& bound_class)
  {
    if constexpr (sizeof...(ARGS) == N)
    {
      bound_class.def(nb::init<ARGS...>());
    }
    else
    {
      // add another FloatT to the list
      defineOpinionNoBaseCtorArgs<FloatT, ARGS...>(bound_class);
    }
  }

  static void defineBinomialDependentFields(::nanobind::class_<sl::OpinionNoBase<N, FloatT>>& bound_class)
    requires sl::is_binomial<N>
  {
    using Opinion = sl::OpinionNoBase<N, FloatT>;
    bound_class.def("belief", nb::overload_cast<>(&Opinion::belief))
        .def("belief", nb::overload_cast<>(&Opinion::belief, nb::const_), nb::rv_policy::reference)
        .def("disbelief", nb::overload_cast<>(&Opinion::disbelief), nb::rv_policy::reference)
        .def("disbelief", nb::overload_cast<>(&Opinion::disbelief, nb::const_), nb::rv_policy::reference)
        .def("complement", &Opinion::complement)
        .def("getBinomialProjection", &Opinion::getBinomialProjection)
        .def("getProbability", &Opinion::getProbability)
        .def("degree_of_conflict", nb::overload_cast<Opinion, FloatT, FloatT>(&Opinion::degree_of_conflict, nb::const_))
        .def("degree_of_harmony", nb::overload_cast<Opinion, FloatT, FloatT>(&Opinion::degree_of_harmony, nb::const_))
        .def("revise_trust_", nb::overload_cast<FloatT, Opinion>(&Opinion::revise_trust_), nb::rv_policy::reference)
        .def("revise_trust_", nb::overload_cast<FloatT>(&Opinion::revise_trust_), nb::rv_policy::reference)
        .def("revise_trust", nb::overload_cast<FloatT, Opinion>(&Opinion::revise_trust, nb::const_))
        .def("revise_trust", nb::overload_cast<FloatT>(&Opinion::revise_trust, nb::const_))
        .def("multiply_", &Opinion::multiply_, nb::rv_policy::reference)
        .def("multiply", &Opinion::multiply)
        .def("comultiply_", &Opinion::comultiply_, nb::rv_policy::reference)
        .def("comultiply", &Opinion::comultiply)
        .def("deduction_", nb::overload_cast<FloatT, Opinion, Opinion>(&Opinion::deduction_), nb::rv_policy::reference)
        .def("deduction", nb::overload_cast<FloatT, Opinion, Opinion>(&Opinion::deduction, nb::const_));
  }

  static void defineBinomialDependentFields(::nanobind::class_<sl::OpinionNoBase<N, FloatT>>& bound_class)
    requires(not sl::is_binomial<N>)
  {
    //  using Opinion = sl::OpinionNoBase<N, FloatT>;
    //  ;
  }

  static void load(::nanobind::module_& bound_module)
  {
    using Opinion = sl::OpinionNoBase<N, FloatT>;
    using BeliefType = typename Opinion::BeliefType;

    std::string module_name{ "OpinionNoBase" };
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
        nb::class_<Opinion>(bound_module, module_name.c_str())
            .def_ro_static("dimension", &Opinion::SIZE)
            .def(nb::init())
            .def(nb::init<BeliefType>(), nb::arg("belief_masses"))
            .def(nb::init<const Opinion&>())
            .def(nb::init<Opinion&&>())
            .def("__deepcopy__", [](const Opinion& a, nb::dict memo) -> Opinion { return a; })
            // access to prop with reference is important!!
            // otherwise, opinion.belief_masses[0] = 1 inside python would not write to the current opinion
            .def_prop_rw(
                "belief_masses",
                [](Opinion& opin) -> BeliefType& { return opin.belief_masses(); },
                [](Opinion& opin, BeliefType belief_masses) { opin.belief_masses() = belief_masses; },
                nb::rv_policy::reference)
            .def("belief_masses_copy", nb::overload_cast<>(&Opinion::belief_masses, nb::const_))
            .def("uncertainty", &Opinion::uncertainty)
            .def("interpolate", &Opinion::interpolate)
            .def_static("NeutralBeliefDistr", &Opinion::NeutralBeliefDistr)
            .def_static("VacuousBeliefDistr", &Opinion::VacuousBeliefDistr)
            .def_static("NeutralBeliefOpinion", &Opinion::NeutralBeliefOpinion)
            .def_static("VacuousBeliefOpinion", &Opinion::VacuousBeliefOpinion)
            .def("dissonance", &Opinion::dissonance)
            .def("getProbabilities", &Opinion::getProbabilities)
            .def("getProjection", nb::overload_cast<BeliefType>(&Opinion::getProjection, nb::const_))
            .def("uncertainty_differential", &Opinion::uncertainty_differential)
            .def("degree_of_conflict",
                 nb::overload_cast<Opinion, BeliefType, BeliefType>(&Opinion::degree_of_conflict, nb::const_))
            .def("degree_of_harmony",
                 nb::overload_cast<Opinion, BeliefType, BeliefType>(&Opinion::degree_of_harmony, nb::const_))
            .def("as_dirichlet", [](Opinion& op) { return static_cast<sl::DirichletDistribution<N, FloatT>>(op); })
            .def("cum_fuse_", &Opinion::cum_fuse_, nb::rv_policy::reference)
            .def("cum_fuse", &Opinion::cum_fuse)
            .def("cum_unfuse_", &Opinion::cum_unfuse_, nb::rv_policy::reference)
            .def("cum_unfuse", &Opinion::cum_unfuse)
            .def("harmony", &Opinion::harmony)
            .def("bc_fuse_", &Opinion::bc_fuse_, nb::rv_policy::reference)
            .def("bc_fuse", &Opinion::bc_fuse)
            .def("average_fuse_", &Opinion::average_fuse_, nb::rv_policy::reference)
            .def("average_fuse", &Opinion::average_fuse)
            .def("wb_fuse_", &Opinion::wb_fuse_, nb::rv_policy::reference)
            .def("wb_fuse", &Opinion::wb_fuse)
            .def("cc_fuse_", &Opinion::cc_fuse_, nb::rv_policy::reference)
            .def("cc_fuse", &Opinion::cc_fuse)
            .def("trust_discount_",
                 nb::overload_cast<sl::OpinionNoBase<2, FloatT>, FloatT>(&Opinion::trust_discount_),
                 nb::rv_policy::reference)
            .def("trust_discount",
                 nb::overload_cast<sl::OpinionNoBase<2, FloatT>, FloatT>(&Opinion::trust_discount, nb::const_))
            .def("trust_discount_", nb::overload_cast<FloatT>(&Opinion::trust_discount_), nb::rv_policy::reference)
            .def("trust_discount", nb::overload_cast<FloatT>(&Opinion::trust_discount, nb::const_))
            .def("limited_trust_discount_",
                 nb::overload_cast<FloatT, sl::OpinionNoBase<2, FloatT>, FloatT>(&Opinion::limited_trust_discount_),
                 nb::rv_policy::reference)
            .def("limited_trust_discount",
                 nb::overload_cast<FloatT, sl::OpinionNoBase<2, FloatT>, FloatT>(&Opinion::limited_trust_discount,
                                                                                 nb::const_))
            .def("limited_trust_discount_",
                 nb::overload_cast<FloatT, FloatT>(&Opinion::limited_trust_discount_),
                 nb::rv_policy::reference)
            .def("limited_trust_discount",
                 nb::overload_cast<FloatT, FloatT>(&Opinion::limited_trust_discount, nb::const_))
            .def("deduction_",
                 nb::overload_cast<BeliefType, sl::Array<N, Opinion>>(&Opinion::deduction_),
                 nb::rv_policy::reference)
            .def("deduction", nb::overload_cast<BeliefType, sl::Array<N, Opinion>>(&Opinion::deduction, nb::const_))
            .def(nb::self == nb::self)
            .def("__repr__", &Opinion::to_string)
            .def("copy", [](const Opinion& opin) -> Opinion { return opin; });

    defineOpinionNoBaseCtorArgs(bound_class);
    defineBinomialDependentFields(bound_class);
  }
};

void loadOpinionNoBaseBindings(::nanobind::module_& bound_module)
{
  loadBindings<OpinionNoBaseLoader>(bound_module);
}
