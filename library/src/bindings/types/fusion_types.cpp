#include "types_bindings.hpp"

#include "subjective_logic_lib/types/fusion_types.hpp"

namespace nb = nanobind;
namespace sl = subjective_logic;

void loadFusionTypes(::nanobind::module_& bound_module)
{
  nb::enum_<sl::FusionType>(bound_module, "FusionType")
      .value("CUMULATIVE", sl::FusionType::CUMULATIVE)
      .value("BELIEF_CONSTRAINT", sl::FusionType::BELIEF_CONSTRAINT)
      .value("AVERAGE", sl::FusionType::AVERAGE)
      .value("WEIGHTED", sl::FusionType::WEIGHTED);
}

void loadRelationTypes(::nanobind::module_& bound_module)
{
  nb::enum_<sl::RelationType>(bound_module, "RelationType")
      .value("HARMONY", sl::RelationType::HARMONY)
      .value("CONFLICT", sl::RelationType::CONFLICT);
}

void loadConflictTypes(::nanobind::module_& bound_module)
{
  nb::enum_<sl::ConflictType>(bound_module, "ConflictType")
      .value("ACCUMULATE", sl::ConflictType::ACCUMULATE)
      .value("AVERAGE", sl::ConflictType::AVERAGE)
      .value("BELIEF_CUMULATIVE", sl::ConflictType::BELIEF_CUMULATIVE)
      .value("BELIEF_BELIEF_CONSTRAINT", sl::ConflictType::BELIEF_BELIEF_CONSTRAINT)
      .value("BELIEF_AVERAGE", sl::ConflictType::BELIEF_AVERAGE)
      .value("BELIEF_WEIGHTED", sl::ConflictType::BELIEF_WEIGHTED);

  bound_module.def("get_belief_fusion_type", &sl::get_belief_fusion_type);
}
void loadTrustRevisionTypes(::nanobind::module_& bound_module)
{
  nb::enum_<sl::TrustRevisionType>(bound_module, "TrustRevisionType")
      .value("NORMAL", sl::TrustRevisionType::NORMAL)
      .value("SHARES", sl::TrustRevisionType::SHARES)
      .value("SHARES_ALLOW_NEGATIVE", sl::TrustRevisionType::SHARES_ALLOW_NEGATIVE)
      .value("REFERENCE_FUSION", sl::TrustRevisionType::REFERENCE_FUSION);
}

void loadAllFusionTypes(::nanobind::module_& bound_module)
{
  loadFusionTypes(bound_module);
  loadRelationTypes(bound_module);
  loadConflictTypes(bound_module);
  loadTrustRevisionTypes(bound_module);
}
