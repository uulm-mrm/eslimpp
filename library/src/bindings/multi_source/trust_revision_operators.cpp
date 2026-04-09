#include "multi_source_bindings.hpp"

#include <string>

#include "subjective_logic_lib/multi_source/trust_revision_operators.hpp"
#include "subjective_logic_lib/util.hpp"

namespace nb = nanobind;
namespace sl = subjective_logic;
namespace slm = subjective_logic::multisource;

template <std::size_t N, typename FloatT>
struct MultiSourceTrustRevisionLoader
{
  static constexpr std::size_t kMaxArgNumber_{ 10 };
  template <typename... ARGS>
  static void loadArbitraryNumberOfArguments(::nanobind::class_<slm::TrustRevision>& bound_class)
    requires(sizeof...(ARGS) > 0)
  {
    bound_class.def_static("revision_factors",
                           nb::overload_cast<sl::RelationType, sl::TrustRevisionType, sl::ConflictType, ARGS...>(
                               &slm::TrustRevision::template revision_factors<ARGS...>));

    // add another ARG to the list
    if constexpr (sizeof...(ARGS) < kMaxArgNumber_)
    {
      loadArbitraryNumberOfArguments<typename sl::FirstType<ARGS...>::type, ARGS...>(bound_class);
    }
  }

  static void load(::nanobind::class_<sl::multisource::TrustRevision>& nb_mod)
  {
    using TOpinion = sl::TrustedOpinion<sl::Opinion<N, FloatT>>;
    using TOpinionNoBase = sl::TrustedOpinion<sl::OpinionNoBase<N, FloatT>>;

    nb_mod.def_static("revision_factors",
                      nb::overload_cast<sl::RelationType,
                                        sl::TrustRevisionType,
                                        sl::ConflictType,
                                        std::vector<TOpinion>,
                                        std::optional<std::vector<bool>>>(
                          &sl::multisource::TrustRevision::template revision_factors<TOpinion>),
                      nb::arg("relation_type"),
                      nb::arg("trust_revision_type"),
                      nb::arg("conflict_type"),
                      nb::arg("opinions"),
                      nb::arg("selection") = nb::none());

    loadArbitraryNumberOfArguments<TOpinion>(nb_mod);
    loadArbitraryNumberOfArguments<TOpinionNoBase>(nb_mod);
  }
};

void loadMultiSourceTrustRevisionOperatorBindings(::nanobind::module_& bound_module)
{
  std::string module_name{ "TrustRevision" };
  auto bound_class = nb::class_<sl::multisource::TrustRevision>(bound_module, module_name.c_str());

  loadBindings<MultiSourceTrustRevisionLoader>(bound_class);
}
