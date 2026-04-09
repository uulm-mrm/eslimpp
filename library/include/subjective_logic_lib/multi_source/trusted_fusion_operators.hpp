#pragma once

// the reader is invited to refer to the following book as reference for the implementations within this file:
// [1] Subjective Logic - A Formalism for Reasoning Under Uncertainty,
// Audun Jøsang, 2016, https://doi.org/10.1007/978-3-319-42337-1
// [2] Multi-source fusion in subjective logic
// A. J⊘sang, D. Wang and J. Zhang,
// 2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017, pp. 1-8,
// doi: 10.23919/ICIF.2017.8009820.
// [3] Multi-Source Fusion Operations in Subjective Logic R. W. Van Der Heijden, H.
// Kopp and F. Kargl, 2018 21st International Conference on Information Fusion (FUSION), Cambridge, UK, 2018, pp.
// 1990-1997, doi: 10.23919/ICIF.2018.8455615.

#include <iostream>
#include <numeric>
#include <vector>
#include <tuple>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/opinions/trusted_opinion.hpp"
#include "subjective_logic_lib/types/fusion_types.hpp"
#include "subjective_logic_lib/multi_source/conflict_operators.hpp"
#include "subjective_logic_lib/multi_source/fusion_operators.hpp"
#include "subjective_logic_lib/multi_source/trust_revision_operators.hpp"

namespace subjective_logic::multisource
{

struct TrustedFusion
{
  // using WeightedTypes = std::tuple<RelationType, TrustRevisionType, ConflictType, double>;
  struct WeightedTypes
  {
    RelationType relation_type;
    TrustRevisionType trust_revision_type;
    ConflictType conflict_type;
    double weight;
  };

  template <typename TrustedOpinionT>
  static inline typename TrustedOpinionT::OpinionT fuse_opinions(FusionType fusion_type,
                                                                 RelationType relation_type,
                                                                 TrustRevisionType trust_revision_type,
                                                                 ConflictType conflict_type,
                                                                 const std::vector<TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <typename TrustedOpinionT>
  static inline typename TrustedOpinionT::OpinionT fuse_opinions(FusionType fusion_type,
                                                                 std::vector<WeightedTypes> weighted_conflict_types,
                                                                 const std::vector<TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <typename TrustedOpinionT>
  static inline typename TrustedOpinionT::OpinionT fuse_opinions(FusionType fusion_type,
                                                                 const std::vector<TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <typename TrustedOpinionT>
  static inline typename TrustedOpinionT::OpinionT fuse_opinions_(FusionType fusion_type,
                                                                  RelationType relation_type,
                                                                  TrustRevisionType trust_revision_type,
                                                                  ConflictType conflict_type,
                                                                  std::vector<TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <typename TrustedOpinionT>
  static inline typename TrustedOpinionT::OpinionT fuse_opinions_(FusionType fusion_type,
                                                                  std::vector<WeightedTypes> weighted_conflict_types,
                                                                  std::vector<TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

protected:
  /**
   * applies the fusion operation and can be used for inplace or copy based operation
   * in order to allow const and non const types, additional template parameter have been added
   * @tparam TrustedOpinionT - trusted opinion type
   * @tparam Vector - either const or non const vector of TrustedOpinion
   * @tparam RevisionFunction - revision function either uses const or non const types
   * @param fusion_type
   * @param weighted_types
   * @param trusted_opinions
   * @param revision_function
   * @return
   */
  template <typename TrustedOpinionT, typename Vector, typename RevisionFunction>
  static inline typename TrustedOpinionT::OpinionT fusion_calculation(FusionType fusion_type,
                                                                      const std::vector<WeightedTypes>& weighted_types,
                                                                      Vector trusted_opinions,
                                                                      RevisionFunction&& revision_function)
    requires is_trusted_opinion<TrustedOpinionT>;

  // in order to allow cuda available generic lambda functions, the function access must be public
  // necessity might vanish, when not using generic lambdas
  // access for test function may be provided using a Test friend class
public:
  /**
   * applies the fusion operation and can be used for inplace or copy based operation
   * in order to allow const and non const types, additional template parameter have been added
   * this function implements a cuda ready implementation with a fixed number of opinions
   * @tparam TrustedOpinionT - trusted opinion type
   * @tparam ArrayT - allows for using const and non const references
   * @tparam RevisionFunction - revision function either uses const or non const types
   * @param fusion_type
   * @param weighted_types
   * @param trusted_opinions
   * @param revision_function
   * @return
   */
  template <std::size_t N, std::size_t M, typename TrustedOpinionT, typename ArrayT, typename RevisionFunction>
  CUDA_AVAIL static inline typename TrustedOpinionT::OpinionT
  fusion_calculation(FusionType fusion_type,
                     const Array<M, WeightedTypes>& weighted_types,
                     ArrayT trusted_opinions,
                     RevisionFunction&& revision_function)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <std::size_t N, std::size_t M, typename TrustedOpinionT>
  CUDA_AVAIL static inline typename TrustedOpinionT::OpinionT
  fuse_opinions(FusionType fusion_type,
                Array<M, WeightedTypes> weighted_conflict_types,
                const Array<N, TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;

  template <std::size_t N, std::size_t M, typename TrustedOpinionT>
  CUDA_AVAIL static inline typename TrustedOpinionT::OpinionT
  fuse_opinions_(FusionType fusion_type,
                 Array<M, WeightedTypes> weighted_conflict_types,
                 Array<N, TrustedOpinionT>& trusted_opinions)
    requires is_trusted_opinion<TrustedOpinionT>;
};

template <typename TrustedOpinionT>
inline typename TrustedOpinionT::OpinionT
TrustedFusion::fuse_opinions(FusionType fusion_type,
                             const RelationType relation_type,
                             const TrustRevisionType trust_revision_type,
                             const ConflictType conflict_type,
                             const std::vector<TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  WeightedTypes types{ relation_type, trust_revision_type, conflict_type, 1.0 };
  return fuse_opinions(fusion_type, { types }, trusted_opinions);
}

template <typename TrustedOpinionT>
inline typename TrustedOpinionT::OpinionT TrustedFusion::fuse_opinions_(FusionType fusion_type,
                                                                        const RelationType relation_type,
                                                                        const TrustRevisionType trust_revision_type,
                                                                        const ConflictType conflict_type,
                                                                        std::vector<TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  WeightedTypes types{ relation_type, trust_revision_type, conflict_type, 1.0 };
  return fuse_opinions_(fusion_type, { types }, trusted_opinions);
}

template <typename TrustedOpinionT>
typename TrustedOpinionT::OpinionT TrustedFusion::fuse_opinions(FusionType fusion_type,
                                                                std::vector<WeightedTypes> weighted_conflict_types,
                                                                const std::vector<TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  auto revision_function = [](const TrustedOpinionT& top, typename TrustedOpinionT::FLOAT_t revision_factor) {
    return top.revise_trust(revision_factor).discounted_opinion();
  };

  return fusion_calculation<TrustedOpinionT, const std::vector<TrustedOpinionT>&>(
      fusion_type, weighted_conflict_types, trusted_opinions, revision_function);
}

template <typename TrustedOpinionT>
typename TrustedOpinionT::OpinionT TrustedFusion::fuse_opinions(FusionType fusion_type,
                                                                const std::vector<TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  auto revision_function = [](const TrustedOpinionT& top, typename TrustedOpinionT::FLOAT_t revision_factor) {
    return top.revise_trust(revision_factor).discounted_opinion();
  };

  return fusion_calculation<TrustedOpinionT, const std::vector<TrustedOpinionT>&>(
      fusion_type, {}, trusted_opinions, revision_function);
}

template <std::size_t N, std::size_t M, typename TrustedOpinionT>
typename TrustedOpinionT::OpinionT TrustedFusion::fuse_opinions(FusionType fusion_type,
                                                                Array<M, WeightedTypes> weighted_conflict_types,
                                                                const Array<N, TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  auto revision_function = [](const TrustedOpinionT& top, typename TrustedOpinionT::FLOAT_t revision_factor) {
    return top.revise_trust(revision_factor).discounted_opinion();
  };

  return fusion_calculation<N, M, TrustedOpinionT>(
      fusion_type, weighted_conflict_types, trusted_opinions, revision_function);
}

template <typename TrustedOpinionT>
typename TrustedOpinionT::OpinionT TrustedFusion::fuse_opinions_(FusionType fusion_type,
                                                                 std::vector<WeightedTypes> weighted_conflict_types,
                                                                 std::vector<TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  auto revision_function = [](TrustedOpinionT& top, typename TrustedOpinionT::FLOAT_t revision_factor) {
    return top.revise_trust_(revision_factor).discounted_opinion();
  };

  return fusion_calculation<TrustedOpinionT, std::vector<TrustedOpinionT>&>(
      fusion_type, weighted_conflict_types, trusted_opinions, revision_function);
}

template <std::size_t N, std::size_t M, typename TrustedOpinionT>
typename TrustedOpinionT::OpinionT TrustedFusion::fuse_opinions_(FusionType fusion_type,
                                                                 Array<M, WeightedTypes> weighted_conflict_types,
                                                                 Array<N, TrustedOpinionT>& trusted_opinions)
  requires is_trusted_opinion<TrustedOpinionT>
{
  auto revision_function = [](TrustedOpinionT& top, typename TrustedOpinionT::FLOAT_t revision_factor) {
    return top.revise_trust_(revision_factor).discounted_opinion();
  };

  return fusion_calculation<N, M, TrustedOpinionT>(
      fusion_type, weighted_conflict_types, trusted_opinions, revision_function);
}

template <typename TrustedOpinionT, typename Vector, typename RevisionFunction>
typename TrustedOpinionT::OpinionT TrustedFusion::fusion_calculation(FusionType fusion_type,
                                                                     const std::vector<WeightedTypes>& weighted_types,
                                                                     Vector trusted_opinions,
                                                                     RevisionFunction&& revision_function)
  requires is_trusted_opinion<TrustedOpinionT>
{
  using FloatT = typename TrustedOpinionT::FloatT;
  using OpinionT = typename TrustedOpinionT::OpinionT;
  const std::size_t number_ops = trusted_opinions.size();

  // in case that the list of types is empty, there is simply no trust revision
  std::vector<FloatT> weighted_revision_factors(number_ops, 0.);
  for (auto [relation_type, trust_revision_type, conflict_type, weight] : weighted_types)
  {
    std::vector<FloatT> revision_factors =
        TrustRevision::revision_factors(relation_type, trust_revision_type, conflict_type, trusted_opinions);

    for (std::size_t op_idx{ 0 }; op_idx < number_ops; ++op_idx)
    {
      weighted_revision_factors[op_idx] += weight * revision_factors[op_idx];
    }
  }

  std::vector<OpinionT> discounted_opinions;
  discounted_opinions.reserve(number_ops);
  std::transform(trusted_opinions.begin(),
                 trusted_opinions.end(),
                 weighted_revision_factors.begin(),
                 std::back_inserter(discounted_opinions),
                 revision_function);

  return Fusion::fuse_opinions(fusion_type, discounted_opinions);
}

template <std::size_t N, std::size_t M, typename TrustedOpinionT, typename ArrayT, typename RevisionFunction>
typename TrustedOpinionT::OpinionT TrustedFusion::fusion_calculation(FusionType fusion_type,
                                                                     const Array<M, WeightedTypes>& weighted_types,
                                                                     ArrayT trusted_opinions,
                                                                     RevisionFunction&& revision_function)
  requires is_trusted_opinion<TrustedOpinionT>
{
  using FloatT = TrustedOpinionT::FloatT;
  using OpinionT = TrustedOpinionT::OpinionT;

  Array<N, FloatT> weighted_revision_factors;
  constexpr_for<0, M>([&](std::size_t idx) {
    auto [relation_type, trust_revision_type, conflict_type, weight] = weighted_types[idx];
    Array<N, FloatT> revision_factors =
        TrustRevision::revision_factors(relation_type, trust_revision_type, conflict_type, trusted_opinions);
    weighted_revision_factors += weight * revision_factors;
  });

  Array<N, OpinionT> discounted_opinions;
  constexpr_for<0, N>([&](std::size_t idx) {
    discounted_opinions[idx] = revision_function(trusted_opinions[idx], weighted_revision_factors[idx]);
  });

  return Fusion::fuse_opinions(fusion_type, discounted_opinions);
}

}  // namespace subjective_logic::multisource
