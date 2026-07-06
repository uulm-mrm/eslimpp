#include "multi_source_bindings.hpp"

#include <string>

#include "subjective_logic_lib/multi_source/fusion_operators.hpp"
#include "subjective_logic_lib/util.hpp"

namespace nb = nanobind;
namespace sl = subjective_logic;
namespace slm = subjective_logic::multisource;

template <std::size_t N, typename FloatT>
struct MultiSourceFusionLoader
{
  static constexpr std::size_t kMaxArgNumber_{ 10 };
  template <typename... ARGS>
  static void loadArbitraryNumberOfArguments(::nanobind::class_<slm::Fusion>& bound_class)
    requires(sizeof...(ARGS) > 0)
  {
    bound_class.def_static("fuse_opinions",
                           nb::overload_cast<sl::FusionType, ARGS...>(&slm::Fusion::template fuse_opinions<ARGS...>));

    // add another ARG to the list
    if constexpr (sizeof...(ARGS) < kMaxArgNumber_)
    {
      loadArbitraryNumberOfArguments<typename sl::FirstType<ARGS...>::type, ARGS...>(bound_class);
    }
  }

  static void load(::nanobind::class_<sl::multisource::Fusion>& nb_mod)
  {
    using Opinion = sl::Opinion<N, FloatT>;
    using OpinionNoBase = sl::OpinionNoBase<N, FloatT>;

    nb_mod.def_static("fuse_opinions",
                      nb::overload_cast<sl::FusionType, std::vector<Opinion>>(
                          &sl::multisource::Fusion::template fuse_opinions<Opinion>));

    loadArbitraryNumberOfArguments<Opinion>(nb_mod);
    loadArbitraryNumberOfArguments<OpinionNoBase>(nb_mod);
  }
};

template <std::size_t N, typename FloatT>
struct SequentialFusionLoader
{
  static void load(::nanobind::class_<sl::multisource::SequentialFusion>& nb_mod)
  {
    using Opinion = sl::Opinion<N, FloatT>;

    nb_mod.def_static("fuse_opinions", &sl::multisource::SequentialFusion::fuse_opinions<Opinion>);
  }
};

void loadMultiSourceFusionOperatorBindings(::nanobind::module_& bound_module)
{
  std::string module_name{ "Fusion" };
  auto bound_class = nb::class_<sl::multisource::Fusion>(bound_module, module_name.c_str());

  loadBindings<MultiSourceFusionLoader>(bound_class);

  std::string module_name_sf{ "SequentialFusion" };
  auto bound_class_sf = nb::class_<sl::multisource::SequentialFusion>(bound_module, module_name_sf.c_str());

  loadBindings<SequentialFusionLoader>(bound_class_sf);
}
