#pragma once

#include <iostream>

namespace subjective_logic
{
enum class FusionType : int
{
  CUMULATIVE = 0,
  BELIEF_CONSTRAINT,
  AVERAGE,
  WEIGHTED,
};

enum class TrustRevisionType : int
{
  NORMAL,
  SHARES,
  SHARES_ALLOW_NEGATIVE,
  REFERENCE_FUSION,
};

enum class RelationType : int
{
  CONFLICT,
  HARMONY
};

enum class ConflictType : int
{
  ACCUMULATE,
  AVERAGE,
  BELIEF_CUMULATIVE,
  BELIEF_BELIEF_CONSTRAINT,
  BELIEF_AVERAGE,
  BELIEF_WEIGHTED,
};

constexpr FusionType get_belief_fusion_type(ConflictType conflict_type)
{
  switch (conflict_type)
  {
    case ConflictType::BELIEF_CUMULATIVE:
    {
      return FusionType::CUMULATIVE;
    }
    case ConflictType::BELIEF_BELIEF_CONSTRAINT:
    {
      return FusionType::BELIEF_CONSTRAINT;
    }
    case ConflictType::BELIEF_AVERAGE:
    {
      return FusionType::AVERAGE;
    }
    case ConflictType::BELIEF_WEIGHTED:
    {
      return FusionType::WEIGHTED;
    }
    default:
    {
      throw std::logic_error{ "Fusion types are only availalbe for Belief Constraint Conflict types, not for: " +
                              std::to_string(static_cast<int>(conflict_type)) };
    }
  }
}

}  // namespace subjective_logic
