#pragma once

// the reader is invited to refer to the following book as reference for the implementations within this file:
// [1] Subjective Logic - A Formalism for Reasoning Under Uncertainty,
// Audun Jøsang, 2016, https://doi.org/10.1007/978-3-319-42337-1

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/opinions/trusted_opinion.hpp"
#include "subjective_logic_lib/multi_source/fusion_operators.hpp"

namespace subjective_logic::multisource
{

struct Conflict
{
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

  static constexpr Fusion::FusionType get_belief_fusion_type(ConflictType conflict_type);

  template <typename OpinionT>
  static inline typename OpinionT::FLOAT_t conflict(ConflictType conflict_type, std::initializer_list<OpinionT> inputs)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  template <typename... Opinions>
  static inline FirstType<Opinions...>::type::FLOAT_t conflict(ConflictType conflict_type, Opinions... opinions)
    requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>;

  template <typename OpinionT>
  static inline typename OpinionT::FLOAT_t conflict(ConflictType conflict_type,
                                                    std::vector<OpinionT> opinions,
                                                    std::optional<std::vector<bool>> use_opinion = std::nullopt)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  template <typename OpinionT>
  static inline typename OpinionT::FLOAT_t harmony(ConflictType conflict_type, std::initializer_list<OpinionT> inputs)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  template <typename... Opinions>
  static inline FirstType<Opinions...>::type::FLOAT_t harmony(ConflictType conflict_type, Opinions... opinions)
    requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>;

  template <typename OpinionT>
  static inline typename OpinionT::FLOAT_t harmony(ConflictType conflict_type,
                                                   std::vector<OpinionT> opinions,
                                                   std::optional<std::vector<bool>> use_opinion = std::nullopt)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * calculates the share to the average conflict of each opiniont
   * @tparam N
   * @tparam typename OpinionT::FLOAT_t
   * @tparam OpinionT
   * @param opinions
   * @return the average conflict and the share of each opinion to that conflict
   */
  template <RelationType RelationT, typename OpinionT>
  static inline std::pair<typename OpinionT::FLOAT_t, std::vector<typename OpinionT::FLOAT_t>>
  conflict_shares(ConflictType conflict_type, std::vector<OpinionT> opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * calculated all components for the belief conflict from josang.
   * the optional reference fusion is only required to implement the specific proposal of the paper....
   * by only considering Opinions here, there is no trust, which would be required.
   * I (wodtko) guess that there is little sense in not using the discounted opinions to get the fusion reference,
   * but the paper suggests it.
   * @tparam OpinionT
   * @param reference_fusion_type
   * @param opinions
   * @param reference_fusion
   * @return
   */
  template <RelationType RelationT, typename OpinionT>
  static typename std::
      tuple<std::vector<typename OpinionT::FLOAT_t>, typename OpinionT::FLOAT_t, typename OpinionT::FLOAT_t>
      belief_conflicts(Fusion::FusionType reference_fusion_type,
                       std::vector<OpinionT> opinions,
                       std::optional<OpinionT> reference_fusion = std::nullopt)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * calculates the uncertainty differentials for each opinion individually
   * @tparam N
   * @tparam typename OpinionT::FLOAT_t
   * @tparam OpinionT
   * @param opinions
   * @return
   */
  template <typename OpinionT>
  static inline std::vector<typename OpinionT::FLOAT_t> uncertainty_differentials(std::vector<OpinionT> opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * calculates the uncertainty differentials for each opinion individually.
   * Here, the trust of each trusted opinion is used.
   * Thus, this function yields the uncertainty differentials (see above) for the trusts of all trusted opinons
   * @tparam TrustedOpinionT
   * @param opinions
   * @return
   */
  template <typename TrustedOpinionT>
  static inline std::vector<typename TrustedOpinionT::OpinionT::FLOAT_t>
  uncertainty_differentials(std::vector<TrustedOpinionT> opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

protected:
  // ConflictOperator is used internally, to type the functions for the specific operator implementation
  template <typename OpinionT>
  using ConflictOperator = std::function<typename OpinionT::FLOAT_t(const std::vector<OpinionT>&)>;

  template <RelationType RelationT, typename OpinionT>
  static inline typename OpinionT::FLOAT_t function_switch(ConflictType conflict_type,
                                                           std::vector<OpinionT> opinions,
                                                           std::optional<std::vector<bool>> use_opinion = std::nullopt)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * accumulated conflict of all "connections" within the given set of opinions used
   * @tparam N - SL opinion dimension
   * @tparam typename OpinionT::FLOAT_t - SL opinion floating point type
   * @tparam OpinionT - SL opinion type depending on the previous two parameters
   * @param opinions - list of SL opinions
   * @param use_opinion - flags which opinions to use
   * @return accumulated conflict
   */
  template <RelationType RelationT, typename OpinionT>
  static typename OpinionT::FLOAT_t accumulated_operator(std::vector<OpinionT> opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * averaged accumulated conflict over all "connections"
   * @tparam N - SL opinion dimension
   * @tparam typename OpinionT::FLOAT_t - SL opinion floating point type
   * @tparam OpinionT - SL opinion type depending on the previous two parameters
   * @param opinions - list of SL opinions
   * @param use_opinion - flags which opinions to use
   * @return averaged conflict
   */
  template <RelationType RelationT, typename OpinionT>
  static typename OpinionT::FLOAT_t average_operator(std::vector<OpinionT> opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * belief conflict using the given fusion operation to create the reference opinion
   * @tparam N - SL opinion dimension
   * @tparam typename OpinionT::FLOAT_t - SL opinion floating point type
   * @tparam OpinionT - SL opinion type depending on the previous two parameters
   * @param opinions - list of SL opinions
   * @param use_opinion - flags which opinions to use
   * @return
   */
  template <RelationType RelationT, typename OpinionT>
  static typename OpinionT::FLOAT_t belief_conflict_operator(Fusion::FusionType reference_fusion_type,
                                                             std::vector<OpinionT> opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;
};

constexpr Fusion::FusionType Conflict::get_belief_fusion_type(ConflictType conflict_type)
{
  switch (conflict_type)
  {
    case ConflictType::BELIEF_CUMULATIVE:
    {
      return Fusion::FusionType::CUMULATIVE;
    }
    case ConflictType::BELIEF_BELIEF_CONSTRAINT:
    {
      return Fusion::FusionType::BELIEF_CONSTRAINT;
    }
    case ConflictType::BELIEF_AVERAGE:
    {
      return Fusion::FusionType::AVERAGE;
    }
    case ConflictType::BELIEF_WEIGHTED:
    {
      return Fusion::FusionType::WEIGHTED;
    }
    default:
    {
      throw std::logic_error{ "Fusion types are only availalbe for Belief Constraint Conflict types, not for: " +
                              std::to_string(static_cast<int>(conflict_type)) };
    }
  }
}
template <Conflict::RelationType RelationT, typename OpinionT>
inline typename OpinionT::FLOAT_t Conflict::function_switch(Conflict::ConflictType conflict_type,
                                                            std::vector<OpinionT> opinions,
                                                            std::optional<std::vector<bool>> use_opinion)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  std::vector<OpinionT> opinions_used;
  if (use_opinion)
  {
    // reserve the max possible number of opinions used
    opinions_used.reserve(opinions.size());

    assert(opinions.size() == (*use_opinion).size());
    for (std::size_t idx{ 0 }; idx < opinions.size(); ++idx)
    {
      if ((*use_opinion)[idx])
      {
        opinions_used.push_back(opinions[idx]);
      }
    }
  }
  else
  {
    opinions_used = std::move(opinions);
  }

  switch (conflict_type)
  {
    case ConflictType::ACCUMULATE:
    {
      return accumulated_operator<RelationT>(opinions_used);
    }
    case ConflictType::AVERAGE:
    {
      return average_operator<RelationT>(opinions_used);
    }
    case ConflictType::BELIEF_CUMULATIVE:
    case ConflictType::BELIEF_BELIEF_CONSTRAINT:
    case ConflictType::BELIEF_AVERAGE:
    case ConflictType::BELIEF_WEIGHTED:
    {
      return belief_conflict_operator<RelationT>(get_belief_fusion_type(conflict_type), opinions_used);
    }
    default:
    {
      throw std::logic_error{ "Conflict calculation is not yet implemented for: " +
                              std::to_string(static_cast<int>(conflict_type)) };
    }
  }
}

template <typename OpinionT>
inline typename OpinionT::FLOAT_t Conflict::conflict(Conflict::ConflictType conflict_type,
                                                     std::initializer_list<OpinionT> inputs)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  return conflict(conflict_type, std::vector<OpinionT>{ inputs });
}

template <typename... Opinions>
inline FirstType<Opinions...>::type::FLOAT_t Conflict::conflict(Conflict::ConflictType conflict_type,
                                                                Opinions... opinions)
  requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>
{
  return conflict(conflict_type, { opinions... });
}

template <typename OpinionT>
inline typename OpinionT::FLOAT_t Conflict::conflict(Conflict::ConflictType conflict_type,
                                                     std::vector<OpinionT> opinions,
                                                     std::optional<std::vector<bool>> use_opinion)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  return function_switch<RelationType::CONFLICT>(conflict_type, opinions, use_opinion);
}

template <typename OpinionT>
inline typename OpinionT::FLOAT_t Conflict::harmony(Conflict::ConflictType conflict_type,
                                                    std::initializer_list<OpinionT> inputs)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  return harmony(conflict_type, std::vector<OpinionT>{ inputs });
}

template <typename... Opinions>
inline FirstType<Opinions...>::type::FLOAT_t Conflict::harmony(Conflict::ConflictType conflict_type,
                                                               Opinions... opinions)
  requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>
{
  return harmony(conflict_type, { opinions... });
}

template <typename OpinionT>
inline typename OpinionT::FLOAT_t Conflict::harmony(Conflict::ConflictType conflict_type,
                                                    std::vector<OpinionT> opinions,
                                                    std::optional<std::vector<bool>> use_opinion)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  return function_switch<RelationType::HARMONY>(conflict_type, opinions, use_opinion);
}

template <Conflict::RelationType RelationT, typename OpinionT>
typename OpinionT::FLOAT_t Conflict::accumulated_operator(std::vector<OpinionT> opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  if (opinions.size() < 2)
  {
    return 0;
  }

  typename OpinionT::FLOAT_t accumulated_conflict{ 0 };
  for (std::size_t idx_outer{ 0 }; idx_outer < opinions.size(); ++idx_outer)
  {
    for (std::size_t idx_inner{ idx_outer + 1 }; idx_inner < opinions.size(); ++idx_inner)
    {
      if constexpr (RelationT == RelationType::CONFLICT)
      {
        accumulated_conflict += opinions[idx_outer].degree_of_conflict(opinions[idx_inner]);
      }
      else
      {
        accumulated_conflict += opinions[idx_outer].degree_of_harmony(opinions[idx_inner]);
      }
    }
  }

  return accumulated_conflict;
}

template <Conflict::RelationType RelationT, typename OpinionT>
typename OpinionT::FLOAT_t Conflict::average_operator(std::vector<OpinionT> opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  std::size_t num_used = opinions.size();
  if (num_used < 2)
  {
    return 0;
  }

  typename OpinionT::FLOAT_t accumulated_conflict = accumulated_operator<RelationT>(opinions);
  // integer division intended, since the number of connections must be an integer
  auto num_connections = static_cast<std::size_t>((num_used * (num_used - 1)) / 2);

  return accumulated_conflict / num_connections;
}

template <Conflict::RelationType RelationT, typename OpinionT>
typename OpinionT::FLOAT_t Conflict::belief_conflict_operator(Fusion::FusionType reference_fusion_type,
                                                              std::vector<OpinionT> opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  typename OpinionT::FLOAT_t avg_conflict =
      std::get<2>(Conflict::belief_conflicts<RelationT>(reference_fusion_type, opinions));
  return avg_conflict;
}

template <Conflict::RelationType RelationT, typename OpinionT>
std::pair<typename OpinionT::FLOAT_t, std::vector<typename OpinionT::FLOAT_t>>
Conflict::conflict_shares(ConflictType conflict_type, std::vector<OpinionT> opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  const std::size_t number_ops = opinions.size();
  double avg_conflict;
  if constexpr (RelationT == RelationType::CONFLICT)
  {
    avg_conflict = Conflict::conflict(conflict_type, opinions);
  }
  else
  {
    avg_conflict = Conflict::harmony(conflict_type, opinions);
  }

  if (avg_conflict < EPS_v<typename OpinionT::FLOAT_t>)
  {
    return { 0., std::vector<typename OpinionT::FLOAT_t>(number_ops, 0.) };
  }

  std::vector<bool> use_opinions(number_ops, true);
  std::vector<typename OpinionT::FLOAT_t> conflict_shares(number_ops, 0.);

  for (std::size_t idx{ 0 }; idx < number_ops; ++idx)
  {
    use_opinions[idx] = false;
    double conflict_wo_self;
    if constexpr (RelationT == RelationType::CONFLICT)
    {
      conflict_wo_self = Conflict::conflict(conflict_type, opinions, use_opinions);
    }
    else
    {
      conflict_wo_self = Conflict::harmony(conflict_type, opinions, use_opinions);
    }
    use_opinions[idx] = true;

    conflict_shares[idx] = 1.0 - conflict_wo_self / avg_conflict;
  }

  return { avg_conflict, conflict_shares };
}

template <Conflict::RelationType RelationT, typename OpinionT>
typename std::tuple<std::vector<typename OpinionT::FLOAT_t>, typename OpinionT::FLOAT_t, typename OpinionT::FLOAT_t>
Conflict::belief_conflicts(Fusion::FusionType reference_fusion_type,
                           std::vector<OpinionT> opinions,
                           std::optional<OpinionT> reference_fusion)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  OpinionT reference;
  if (reference_fusion)
  {
    reference = *reference_fusion;
  }
  else
  {
    reference = Fusion::fuse_opinions(reference_fusion_type, opinions);
  }

  std::vector<typename OpinionT::FLOAT_t> conflicts;
  conflicts.reserve(opinions.size());
  for (auto const& opinion : opinions)
  {
    typename OpinionT::FLOAT_t reference_conflict;
    if constexpr (RelationT == RelationType::CONFLICT)
    {
      reference_conflict = reference.degree_of_conflict(opinion);
    }
    else
    {
      reference_conflict = reference.degree_of_harmony(opinion);
    }
    conflicts.push_back(reference_conflict);
  }

  typename OpinionT::FLOAT_t max_conflict{ 0. };
  typename OpinionT::FLOAT_t acc_conflict{ 0. };
  for (typename OpinionT::FLOAT_t conflict : conflicts)
  {
    if (conflict > max_conflict)
    {
      max_conflict = conflict;
    }
    acc_conflict += conflict;
  }
  typename OpinionT::FLOAT_t avg_conflict = acc_conflict / opinions.size();

  return { conflicts, max_conflict, avg_conflict };
}

template <typename OpinionT>
std::vector<typename OpinionT::FLOAT_t> Conflict::uncertainty_differentials(std::vector<OpinionT> opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  double sum_of_uncertainty =
      std::accumulate(opinions.begin(), opinions.end(), 0., [](typename OpinionT::FLOAT_t sum, OpinionT ops) {
        return sum + ops.uncertainty();
      });

  if (sum_of_uncertainty < EPS_v<typename OpinionT::FLOAT_t>)
  {
    return std::vector<typename OpinionT::FLOAT_t>(opinions.size(), 0.);
  }

  std::vector<typename OpinionT::FLOAT_t> differentials;
  differentials.reserve(opinions.size());

  std::transform(opinions.begin(),
                 opinions.end(),
                 std::back_inserter(differentials),
                 [sum_of_uncertainty](OpinionT ops) { return ops.uncertainty() / sum_of_uncertainty; });

  return differentials;
}

template <typename TrustedOpinionT>
std::vector<typename TrustedOpinionT::OpinionT::FLOAT_t>
Conflict::uncertainty_differentials(std::vector<TrustedOpinionT> opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  return uncertainty_differentials(TrustedOpinionT::extractTrusts(opinions));
}

}  // namespace subjective_logic::multisource
