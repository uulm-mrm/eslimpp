#include "multi_source_bindings.hpp"

#include <string>
#include "subjective_logic_lib/multi_source/conflict_operators.hpp"

namespace nb = nanobind;
namespace sl = subjective_logic;
namespace slm = subjective_logic::multisource;

template <std::size_t N, typename FloatT>
struct MultiSourceConflictLoader
{
  static constexpr std::size_t kMaxArgNumber_{ 10 };
  template <typename... ARGS>
  static void loadArbitraryNumberOfArguments(::nanobind::class_<sl::multisource::Conflict>& bound_class)
    requires(sizeof...(ARGS) > 0)
  {
    bound_class.def_static(
        "conflict",
        nb::overload_cast<slm::Conflict::ConflictType, ARGS...>(&slm::Conflict::template conflict<ARGS...>));

    // add another ARG to the list
    if constexpr (sizeof...(ARGS) < kMaxArgNumber_)
    {
      loadArbitraryNumberOfArguments<typename sl::FirstType<ARGS...>::type, ARGS...>(bound_class);
    }
  }

  static void load(::nanobind::class_<sl::multisource::Conflict>& nb_mod)
  {
    using Opinion = sl::Opinion<N, FloatT>;
    using OpinionNoBase = sl::OpinionNoBase<N, FloatT>;

    //    nb_mod.def_static(
    //        "conflict",
    //        nb::overload_cast<slm::Conflict::ConflictType, std::vector<Opinion>, std::optional<std::vector<bool>>>(
    //            &slm::Conflict::template conflict<Opinion>));
    nb_mod.def_static("conflict", [](slm::Conflict::ConflictType conflict_type, std::vector<Opinion> vec) -> FloatT {
      return slm::Conflict::conflict<Opinion>(conflict_type, vec);
    });
    nb_mod.def_static(
        "conflict_selective",
        [](slm::Conflict::ConflictType conflict_type, std::vector<Opinion> vec, std::vector<bool> vec_bool) -> FloatT {
          return slm::Conflict::conflict<Opinion>(conflict_type, vec, vec_bool);
        });

    loadArbitraryNumberOfArguments<Opinion>(nb_mod);
    loadArbitraryNumberOfArguments<OpinionNoBase>(nb_mod);
  }
};

void loadConflictTypes(::nanobind::class_<sl::multisource::Conflict>& nb_mod)
{
  nb::enum_<slm::Conflict::ConflictType>(nb_mod, "ConflictType")
      .value("ACCUMULATE", slm::Conflict::ConflictType::ACCUMULATE)
      .value("AVERAGE", slm::Conflict::ConflictType::AVERAGE)
      .value("BELIEF_CUMULATIVE", slm::Conflict::ConflictType::BELIEF_CUMULATIVE)
      .value("BELIEF_BELIEF_CONSTRAINT", slm::Conflict::ConflictType::BELIEF_BELIEF_CONSTRAINT)
      .value("BELIEF_AVERAGE", slm::Conflict::ConflictType::BELIEF_AVERAGE)
      .value("BELIEF_WEIGHTED", slm::Conflict::ConflictType::BELIEF_WEIGHTED);

  nb_mod.def_static("get_belief_fusion_type", &slm::Conflict::get_belief_fusion_type);
}

void loadMultiSourceConflictOperatorBindings(::nanobind::module_& bound_module)
{
  std::string module_name{ "Conflict" };
  auto bound_class = nb::class_<sl::multisource::Conflict>(bound_module, module_name.c_str());

  loadConflictTypes(bound_class);

  loadBindings<MultiSourceConflictLoader>(bound_class);
}
