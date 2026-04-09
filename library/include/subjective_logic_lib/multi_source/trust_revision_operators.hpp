#pragma once

// the reader is invited to refer to the following book as reference for the implementations within this file:
// [1] Subjective Logic - A Formalism for Reasoning Under Uncertainty,
// Audun Jøsang, 2016, https://doi.org/10.1007/978-3-319-42337-1

#include <iostream>
#include <numeric>
#include <vector>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/types/fusion_types.hpp"
#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/opinions/trusted_opinion.hpp"
#include "subjective_logic_lib/multi_source/fusion_operators.hpp"
#include "subjective_logic_lib/multi_source/conflict_operators.hpp"

#define BELIEF_REVISION_FOLLOWING_JOSAN 1

namespace subjective_logic::multisource
{

struct TrustRevision
{
  template <typename TrustedOpinionT>
  static inline std::vector<typename TrustedOpinionT::FLOAT_t>
  revision_factors(RelationType relation_type,
                   TrustRevisionType trust_revision_type,
                   ConflictType conflict_type,
                   std::initializer_list<TrustedOpinionT> inputs)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <typename... Opinions>
  static inline std::vector<typename FirstType<Opinions...>::type::FLOAT_t>
  revision_factors(RelationType relation_type,
                   TrustRevisionType trust_revision_type,
                   ConflictType conflict_type,
                   Opinions... opinions)
    requires is_trusted_opinion_list<Opinions...> and (sizeof...(Opinions) > 0);

  template <typename TrustedOpinionT>
  static inline std::vector<typename TrustedOpinionT::FLOAT_t>
  revision_factors(RelationType relation_type,
                   TrustRevisionType trust_revision_type,
                   ConflictType conflict_type,
                   std::vector<TrustedOpinionT> opinions,
                   std::optional<std::vector<bool>> use_opinion = std::nullopt)
    requires is_trusted_opinion<TrustedOpinionT>;

protected:
  // RevisionOperator is used internally, to type the functions for the specific operator implementation
  template <typename TrustedOpinionT>
  using RevisionOperator =
      std::function<std::vector<typename TrustedOpinionT::FLOAT_t>(ConflictType conflict_type,
                                                                   const std::vector<TrustedOpinionT>&)>;

  /**
   * Todo add source of own paper
   * calculates revision factors as given in [1] extended for the multi-source case as proposed in [OWN PAPER] using the
   * accumulated conflict
   * @tparam TrustedOpinionT
   * @param opinions
   * @return
   */
  template <typename TrustedOpinionT>
  static inline std::vector<typename TrustedOpinionT::FLOAT_t>
  normal_trust_revision(RelationType relation_type,
                        ConflictType conflict_type,
                        const std::vector<TrustedOpinionT>& opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <typename TrustedOpinionT>
  static inline std::vector<typename TrustedOpinionT::FLOAT_t>
  conflict_shares_trust_revision(RelationType relation_type,
                                 ConflictType conflict_type,
                                 const std::vector<TrustedOpinionT>& opinions,
                                 bool positive_scores_only = false)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <typename TrustedOpinionT>
  static inline std::vector<typename TrustedOpinionT::FLOAT_t>
  reference_fusion_trust_revision(RelationType relation_type,
                                  ConflictType conflict_type,
                                  const std::vector<TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

  // in order to allow cuda available generic lambda functions, the function access must be public
  // necessity might vanish, when not using generic lambdas
  // access for test function may be provided using a Test friend class
public:
  template <std::size_t N, typename TrustedOpinionT>
  CUDA_AVAIL static inline Array<N, typename TrustedOpinionT::FLOAT_t>
  conflict_shares_trust_revision(RelationType relation_type,
                                 ConflictType conflict_type,
                                 const Array<N, TrustedOpinionT>& opinions,
                                 bool positive_scores_only = false)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <std::size_t N, typename TrustedOpinionT>
  CUDA_AVAIL static inline Array<N, typename TrustedOpinionT::FLOAT_t>
  revision_factors(RelationType relation_type,
                   TrustRevisionType trust_revision_type,
                   ConflictType conflict_type,
                   Array<N, TrustedOpinionT> opinions)
    requires is_trusted_opinion<TrustedOpinionT>;
};

template <typename TrustedOpinionT>
inline std::vector<typename TrustedOpinionT::FLOAT_t>
TrustRevision::revision_factors(RelationType relation_type,
                                TrustRevisionType trust_revision_type,
                                ConflictType conflict_type,
                                std::initializer_list<TrustedOpinionT> inputs)
  requires is_trusted_opinion<TrustedOpinionT>
{
  return TrustRevision::revision_factors(
      relation_type, trust_revision_type, conflict_type, std::vector<TrustedOpinionT>{ inputs });
}

template <typename... Opinions>
inline std::vector<typename FirstType<Opinions...>::type::FLOAT_t>
TrustRevision::revision_factors(RelationType relation_type,
                                TrustRevisionType trust_revision_type,
                                ConflictType conflict_type,
                                Opinions... opinions)
  requires is_trusted_opinion_list<Opinions...> and (sizeof...(Opinions) > 0)
{
  return TrustRevision::revision_factors(relation_type, trust_revision_type, conflict_type, { opinions... });
}

template <typename TrustedOpinionT>
inline std::vector<typename TrustedOpinionT::FLOAT_t>
TrustRevision::revision_factors(RelationType relation_type,
                                TrustRevisionType trust_revision_type,
                                ConflictType conflict_type,
                                std::vector<TrustedOpinionT> opinions,
                                std::optional<std::vector<bool>> use_opinion)
  requires is_trusted_opinion<TrustedOpinionT>
{
  std::vector<TrustedOpinionT> opinions_used;
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

  switch (trust_revision_type)
  {
    case TrustRevisionType::NORMAL:
    {
      return normal_trust_revision<TrustedOpinionT>(relation_type, conflict_type, opinions_used);
    }
    case TrustRevisionType::SHARES:
    {
      return conflict_shares_trust_revision<TrustedOpinionT>(relation_type, conflict_type, opinions_used, true);
    }
    case TrustRevisionType::SHARES_ALLOW_NEGATIVE:
    {
      return conflict_shares_trust_revision<TrustedOpinionT>(relation_type, conflict_type, opinions_used, false);
    }
    case TrustRevisionType::REFERENCE_FUSION:
    {
      return reference_fusion_trust_revision<TrustedOpinionT>(relation_type, conflict_type, opinions_used);
    }
    default:
    {
      throw std::logic_error{ "TrustRevision is not yet implemented for: " +
                              std::to_string(static_cast<int>(trust_revision_type)) };
    }
  }
}

template <std::size_t N, typename TrustedOpinionT>
inline Array<N, typename TrustedOpinionT::FLOAT_t>
TrustRevision::revision_factors(const RelationType relation_type,
                                const TrustRevisionType trust_revision_type,
                                const ConflictType conflict_type,
                                Array<N, TrustedOpinionT> opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  switch (trust_revision_type)
  {
    // case TrustRevisionType::NORMAL:
    // {
    //   return normal_trust_revision<TrustedOpinionT>(relation_type, conflict_type, opinions_used);
    // }
    case TrustRevisionType::SHARES:
    {
      return conflict_shares_trust_revision<N, TrustedOpinionT>(relation_type, conflict_type, opinions, true);
    }
    case TrustRevisionType::SHARES_ALLOW_NEGATIVE:
    {
      return conflict_shares_trust_revision<N, TrustedOpinionT>(relation_type, conflict_type, opinions, false);
    }
    // case trustrevisiontype::reference_fusion:
    // {
    //   return reference_fusion_trust_revision<trustedopiniont>(relation_type, conflict_type, opinions_used);
    // }
    default:
    {
      printf("TrustRevision is not yet implemented");
      return Array<N, typename TrustedOpinionT::FLOAT_t>{};
    }
  }
}

template <typename TrustedOpinionT>
inline std::vector<typename TrustedOpinionT::FLOAT_t>
TrustRevision::normal_trust_revision(const RelationType relation_type,
                                     ConflictType conflict_type,
                                     const std::vector<TrustedOpinionT>& opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  using FloatT = TrustedOpinionT::FLOAT_t;
  std::size_t num_ops{ opinions.size() };

  std::vector<typename TrustedOpinionT::OpinionT> discounted_opinions =
      TrustedOpinionT::extractDiscountedOpinions(opinions);

  std::vector<FloatT> uncertainty_differentials = Conflict::uncertainty_differentials(opinions);
  FloatT conflict;
  if (relation_type == RelationType::CONFLICT)
  {
    conflict = Conflict::conflict(conflict_type, discounted_opinions);
  }
  else
  {
    conflict = Conflict::harmony(conflict_type, discounted_opinions);
  }

  std::vector<FloatT> revision_factors;
  revision_factors.reserve(num_ops);
  for (std::size_t idx{ 0 }; idx < num_ops; ++idx)
  {
    FloatT score = uncertainty_differentials[idx] * conflict;
    if (relation_type == RelationType::HARMONY)
    {
      score *= -1;
    }
    revision_factors.push_back(score);
  }

  return revision_factors;
}

template <typename TrustedOpinionT>
inline std::vector<typename TrustedOpinionT::FLOAT_t>
TrustRevision::conflict_shares_trust_revision(const RelationType relation_type,
                                              ConflictType conflict_type,
                                              const std::vector<TrustedOpinionT>& opinions,
                                              bool positive_scores_only)
  requires is_trusted_opinion<TrustedOpinionT>
{
  using FloatT = TrustedOpinionT::FLOAT_t;
  std::size_t num_ops{ opinions.size() };

  auto raw_opinions = TrustedOpinionT::extractOpinions(opinions);
  std::vector<typename TrustedOpinionT::OpinionT> discounted_opinions =
      TrustedOpinionT::extractDiscountedOpinions(opinions);

  // it only makes sense to distribute conflict based on average fusion; thus conflict_shares are calculated using
  // AVERAGE if, however, the demanded conflict type differs, the absolute overall conflict is calculated with the
  // respective conflict type

  //  auto [conflict, conflict_shares] = Conflict::conflict_shares<RelationT>(ConflictType::AVERAGE,
  //  discounted_opinions);
  auto [conflict, conflict_shares] = Conflict::conflict_shares(relation_type, ConflictType::AVERAGE, raw_opinions);
  if (conflict_type != ConflictType::AVERAGE)
  {
    if (relation_type == RelationType::CONFLICT)
    {
      conflict = Conflict::conflict(conflict_type, discounted_opinions);
    }
    else
    {
      conflict = Conflict::harmony(conflict_type, discounted_opinions);
    }
  }

  std::vector<FloatT> revision_factors;
  revision_factors.reserve(num_ops);
  for (std::size_t idx{ 0 }; idx < num_ops; ++idx)
  {
    FloatT share = conflict_shares[idx];
    if (positive_scores_only and share < 0)
    {
      revision_factors.push_back(0);
      continue;
    }
    FloatT scaled_share = conflict * share;
    if (relation_type == RelationType::HARMONY)
    {
      scaled_share *= -1;
    }
    revision_factors.push_back(scaled_share);
  }

  return revision_factors;
}

template <std::size_t N, typename TrustedOpinionT>
inline Array<N, typename TrustedOpinionT::FLOAT_t>
TrustRevision::conflict_shares_trust_revision(const RelationType relation_type,
                                              ConflictType conflict_type,
                                              const Array<N, TrustedOpinionT>& opinions,
                                              bool positive_scores_only)
  requires is_trusted_opinion<TrustedOpinionT>
{
  using FloatT = TrustedOpinionT::FLOAT_t;

  auto raw_opinions = TrustedOpinionT::extractOpinions(opinions);
  auto discounted_opinions = TrustedOpinionT::extractDiscountedOpinions(opinions);

  // it only makes sense to distribute conflict based on average fusion; thus conflict_shares are calculated using
  // AVERAGE if, however, the demanded conflict type differs, the absolute overall conflict is calculated with the
  // respective conflict type

  FloatT conflict{ 0. };
  Array<N, FloatT> conflict_shares =
      Conflict::conflict_shares(relation_type, ConflictType::AVERAGE, raw_opinions, conflict);
  if (conflict_type != ConflictType::AVERAGE)
  {
    if (relation_type == RelationType::CONFLICT)
    {
      conflict = Conflict::conflict(conflict_type, discounted_opinions);
    }
    else
    {
      conflict = Conflict::harmony(conflict_type, discounted_opinions);
    }
  }

  Array<N, FloatT> revision_factors;
  constexpr_for<0, N>([&, conflict](std::size_t idx) {
    FloatT share = conflict_shares[idx];
    if (positive_scores_only and share < 0)
    {
      revision_factors[idx] = 0;
      return;
    }
    FloatT scaled_share{ conflict * share };
    if (relation_type == RelationType::HARMONY)
    {
      scaled_share *= -1;
    }
    revision_factors[idx] = scaled_share;
  });

  return revision_factors;
}

template <typename TrustedOpinionT>
inline std::vector<typename TrustedOpinionT::FLOAT_t>
TrustRevision::reference_fusion_trust_revision(RelationType relation_type,
                                               ConflictType conflict_type,
                                               const std::vector<TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  using FloatT = TrustedOpinionT::FLOAT_t;
  std::size_t num_ops{ trusted_opinions.size() };

  auto discounted_opinions = TrustedOpinionT::extractDiscountedOpinions(trusted_opinions);

#ifdef BELIEF_REVISION_FOLLOWING_JOSAN
  auto opinions = TrustedOpinionT::extractOpinions(trusted_opinions);
  // use the discounted opinions for fusion here
  typename TrustedOpinionT::OpinionT reference_fusion =
      Fusion::fuse_opinions(get_belief_fusion_type(conflict_type), discounted_opinions);
  auto [belief_conflicts, max_conflict, avg_conflict] =
      Conflict::belief_conflicts(relation_type, get_belief_fusion_type(conflict_type), opinions, { reference_fusion });
#else
  auto [belief_conflicts, max_conflict, avg_conflict] =
      Conflict::belief_conflicts<RelationT>(Conflict::get_belief_fusion_type(conflict_type), discounted_opinions);
#endif

  FloatT denom = max_conflict - avg_conflict;

  std::vector<FloatT> revision_factors;
  revision_factors.reserve(num_ops);
  for (std::size_t idx{ 0 }; idx < num_ops; ++idx)
  {
    FloatT relative_conflict = belief_conflicts[idx] - avg_conflict;

    if (relative_conflict <= 0)
    {
      revision_factors.push_back(0);
      continue;
    }
    FloatT revision_factor = max_conflict * relative_conflict / denom;

    if (relation_type == RelationType::HARMONY)
    {
      revision_factor *= -1;
    }
    revision_factors.push_back(revision_factor);
  }

  return revision_factors;
}

}  // namespace subjective_logic::multisource