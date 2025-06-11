#include "multi_source_bindings.hpp"

#include <string>

#include <nanobind/stl/tuple.h>
#include "subjective_logic_lib/multi_source/trusted_fusion_operators.hpp"

namespace nb = nanobind;
namespace sl = subjective_logic;
namespace slm = subjective_logic::multisource;

template <std::size_t N, typename FloatT>
struct MultiSourceTrustedFusionLoader
{
  template <typename OpinionT>
  static void loadTrusted(::nanobind::class_<sl::multisource::TrustedFusion>& nb_mod)
  {
    using TrustedOpinionT = sl::TrustedOpinion<OpinionT>;
    nb_mod.def_static("fuse_opinions",
                      nb::overload_cast<slm::Fusion::FusionType, const std::vector<TrustedOpinionT>&>(
                          &sl::multisource::TrustedFusion::template fuse_opinions<TrustedOpinionT>));

    nb_mod.def_static("fuse_opinions",
                      nb::overload_cast<slm::Fusion::FusionType,
                                        slm::TrustRevision::TrustRevisionType,
                                        slm::Conflict::ConflictType,
                                        const std::vector<TrustedOpinionT>&>(
                          &sl::multisource::TrustedFusion::template fuse_opinions<TrustedOpinionT>));

    // by explicitly defining the function call, the result is made available in python, since nanobind does not seem to
    // support input/output parameter
    nb_mod.def_static("fuse_opinions_",
                      [](slm::Fusion::FusionType fusion_type,
                         slm::TrustRevision::TrustRevisionType revision_type,
                         slm::Conflict::ConflictType conflict_type,
                         std::vector<TrustedOpinionT>& t_vec) {
                        auto fusion = slm::TrustedFusion::fuse_opinions_<TrustedOpinionT>(
                            fusion_type, revision_type, conflict_type, t_vec);
                        return std::make_tuple(fusion, t_vec);
                      });

    nb_mod.def_static("fuse_opinions",
                      nb::overload_cast<slm::Fusion::FusionType,
                                        std::vector<slm::TrustedFusion::WeightedTypes>,
                                        const std::vector<TrustedOpinionT>&>(
                          &sl::multisource::TrustedFusion::template fuse_opinions<TrustedOpinionT>));

    // by explicitly defining the function call, the result is made available in python, since nanobind does not seem to
    // support input/output parameter
    nb_mod.def_static("fuse_opinions_",
                      [](slm::Fusion::FusionType fusion_type,
                         std::vector<slm::TrustedFusion::WeightedTypes> types_vec,
                         std::vector<TrustedOpinionT>& t_vec) {
                        auto fusion =
                            slm::TrustedFusion::fuse_opinions_<TrustedOpinionT>(fusion_type, types_vec, t_vec);
                        return std::make_tuple(fusion, t_vec);
                      });
  }

  static void load(::nanobind::class_<sl::multisource::TrustedFusion>& nb_mod)
  {
    using Opinion = sl::Opinion<N, FloatT>;
    using OpinionNoBase = sl::OpinionNoBase<N, FloatT>;

    loadTrusted<Opinion>(nb_mod);
    loadTrusted<OpinionNoBase>(nb_mod);
  }
};

void loadMultiSourceTrustedFusionOperatorBindings(::nanobind::module_& bound_module)
{
  std::string module_name{ "TrustedFusion" };
  auto bound_class = nb::class_<sl::multisource::TrustedFusion>(bound_module, module_name.c_str());

  loadBindings<MultiSourceTrustedFusionLoader>(bound_class);
}
