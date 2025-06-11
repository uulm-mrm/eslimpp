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
#include <optional>
#include <tuple>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/opinions/opinion.hpp"

namespace subjective_logic::multisource
{

struct Fusion
{
  enum class FusionType : int
  {
    CUMULATIVE = 0,
    BELIEF_CONSTRAINT,
    AVERAGE,
    WEIGHTED,
  };

  template <typename OpinionT>
  static inline OpinionT fuse_opinions(FusionType fusion_type, std::vector<OpinionT> opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  template <typename OpinionT>
  static inline OpinionT fuse_opinions(FusionType fusion_type, std::initializer_list<OpinionT>& inputs)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  template <typename... Opinions>
  static inline FirstType<Opinions...>::type fuse_opinions(FusionType fusion_type, Opinions... opinions)
    requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>;

protected:
  // FusionOperator is used internally, to type the functions for the specific operator implementation
  template <typename OpinionT>
  using FusionOperator = std::function<
      OpinionT(const std::vector<OpinionT>&, std::vector<typename OpinionT::FLOAT_t>, typename OpinionT::FLOAT_t)>;

  /**
   * preprocessing steps of all multi source fusion operators are combined in this function
   * in case that some opinions are dogmatic, the result is almost always just a mean of all dogmatic opinions,
   * thus, in this case, the result is precalculated an returned by this function
   * @tparam N
   * @tparam FloatT
   * @param opinions
   * @return
   */
  template <typename OpinionT>
  static inline std::tuple<std::vector<typename OpinionT::FLOAT_t>, std::optional<OpinionT>>
  preprocess_opinions(const std::vector<OpinionT>& opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * the prior is not handled by any multi source fusion model, instead it gets averaged over all available opinions
   * @tparam OpinionT
   * @param opinions
   * @return
   */
  template <typename OpinionT>
  static inline typename OpinionT::BeliefType average_prior(const std::vector<OpinionT>& opinions)
    requires is_opinion<OpinionT>;

  /**
   * fuse all opinions using a given fusion operator
   * this functions handles everything except the actual fusion operation
   * the given fusion operator is provided with precalculated uncertainties and the uncertainty product
   * further, using the preprocess_opinions function, dogmatic opinions are handled.
   * @tparam N
   * @tparam FloatT
   * @param opinions
   * @return
   */
  template <typename OpinionT>
  static inline OpinionT fuse_opinions_(const std::vector<OpinionT>& opinions, FusionOperator<OpinionT> fusion_operator)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * includes all operator specific calculations and is used together with fuse_opinions
   * @tparam N
   * @tparam FloatT
   * @tparam OpinionT
   * @param opinions
   * @param uncertainties
   * @param uncertainty_product
   * @return
   */
  template <typename OpinionT>
  static inline OpinionT cumulative_fusion_operator(const std::vector<OpinionT>& opinions,
                                                    std::vector<typename OpinionT::FLOAT_t> uncertainties,
                                                    typename OpinionT::FLOAT_t uncertainty_product)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * includes all operator specific calculations and is used together with fuse_opinions
   * @tparam N
   * @tparam FloatT
   * @tparam OpinionT
   * @param opinions
   * @param uncertainties
   * @param uncertainty_product
   * @return
   */
  template <typename OpinionT>
  static inline OpinionT belief_constraint_fusion_operator(const std::vector<OpinionT>& opinions,
                                                           std::vector<typename OpinionT::FLOAT_t> uncertainties,
                                                           typename OpinionT::FLOAT_t uncertainty_product)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * includes all operator specific calculations and is used together with fuse_opinions
   * @tparam N
   * @tparam FloatT
   * @tparam OpinionT
   * @param opinions
   * @param uncertainties
   * @param uncertainty_product
   * @return
   */
  template <typename OpinionT>
  static inline OpinionT average_fusion_operator(const std::vector<OpinionT>& opinions,
                                                 std::vector<typename OpinionT::FLOAT_t> uncertainties,
                                                 typename OpinionT::FLOAT_t uncertainty_product)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;
};

template <typename OpinionT>
inline OpinionT Fusion::fuse_opinions(Fusion::FusionType fusion_type, std::initializer_list<OpinionT>& inputs)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  return fuse_opinions(fusion_type, std::vector<OpinionT>{ inputs });
}

template <typename... Opinions>
inline FirstType<Opinions...>::type Fusion::fuse_opinions(Fusion::FusionType fusion_type, Opinions... opinions)
  requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>
{
  using OutType = FirstType<Opinions...>::type;
  return OutType{ fuse_opinions(fusion_type, std::vector<OutType>{ opinions... }) };
}

template <typename OpinionT>
inline OpinionT Fusion::fuse_opinions(Fusion::FusionType fusion_type, std::vector<OpinionT> opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  switch (fusion_type)
  {
    case FusionType::CUMULATIVE:
    {
      return fuse_opinions_(opinions, Fusion::cumulative_fusion_operator<OpinionT>);
    }
    case FusionType::BELIEF_CONSTRAINT:
    {
      return fuse_opinions_(opinions, Fusion::belief_constraint_fusion_operator<OpinionT>);
    }
    case FusionType::AVERAGE:
    {
      return fuse_opinions_(opinions, Fusion::average_fusion_operator<OpinionT>);
    }
    default:
    {
      throw std::logic_error{ "MultiSource fusion is not yet implemented for: " +
                              std::to_string(static_cast<int>(fusion_type)) };
    }
  }
}

template <typename OpinionT>
inline typename OpinionT::BeliefType Fusion::average_prior(const std::vector<OpinionT>& opinions)
  requires is_opinion<OpinionT>
{
  typename OpinionT::BeliefType prior{ 0 };

  for (auto const& opinion : opinions)
  {
    for (std::size_t idx{ 0 }; idx < OpinionT::SIZE; ++idx)
    {
      prior[idx] += opinion.prior_belief_masses()[idx];
    }
  }
  for (std::size_t idx{ 0 }; idx < OpinionT::SIZE; ++idx)
  {
    prior[idx] /= opinions.size();
  }

  return prior;
}

template <typename OpinionT>
inline std::tuple<std::vector<typename OpinionT::FLOAT_t>, std::optional<OpinionT>>
Fusion::preprocess_opinions(const std::vector<OpinionT>& opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  using FloatT = typename OpinionT::FLOAT_t;
  if (opinions.size() == 1)
  {
    return std::make_tuple(std::vector<typename OpinionT::FLOAT_t>{}, *opinions.begin());
  }

  const std::size_t n_elements = opinions.size();
  std::vector<FloatT> uncertainties(opinions.size());
  std::transform(
      opinions.begin(), opinions.end(), uncertainties.begin(), [](OpinionT opinion) { return opinion.uncertainty(); });

  // search for dogmatic opinions to use different fusion approach if necessary
  bool near_zero_uncertainties{ false };
  std::vector<OpinionT> dogmatic_opinions;
  for (std::size_t idx{ 0 }; idx < n_elements; ++idx)
  {
    if (std::abs(uncertainties[idx]) < EPS_v<typename OpinionT::FLOAT_t>)
    {
      near_zero_uncertainties = true;
      dogmatic_opinions.push_back(opinions[idx]);
    }
  }

  if (near_zero_uncertainties)
  {
    OpinionT result;
    // consider all opinions with near zero uncertainties as equally strong dogmatic opinions
    // (meaning that the mean is calculated instead of separately consider the limes as given in [2])
    for (const auto& opinion : dogmatic_opinions)
    {
      std::transform(result.belief_masses().begin(),
                     result.belief_masses().end(),
                     opinion.belief_masses().begin(),
                     result.belief_masses().begin(),
                     [](FloatT val1, FloatT val2) { return val1 + val2; });
    }
    const std::size_t n_dogmatic_elements = dogmatic_opinions.size();
    std::transform(result.belief_masses().begin(),
                   result.belief_masses().end(),
                   result.belief_masses().begin(),
                   [n_dogmatic_elements](FloatT val1) { return val1 / static_cast<FloatT>(n_dogmatic_elements); });
    return std::make_tuple(std::vector<FloatT>{}, result);
  }
  return std::make_tuple(std::move(uncertainties), std::nullopt);
}

template <typename OpinionT>
OpinionT Fusion::fuse_opinions_(const std::vector<OpinionT>& opinions, FusionOperator<OpinionT> fusion_operator)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  using FloatT = typename OpinionT::FLOAT_t;
  auto [uncertainties, pre_result] = Fusion::preprocess_opinions(opinions);
  if (pre_result)
  {
    return *pre_result;
  }

  // the product of all uncertainties can later be used to calculate specific products
  // by dividing the respective uncertainty which is omitted
  const FloatT uncert_prod = std::accumulate(uncertainties.begin(), uncertainties.end(), 1., std::multiplies<FloatT>());

  OpinionT result = fusion_operator(opinions, uncertainties, uncert_prod);
  if constexpr (is_opinion<OpinionT>)
  {
    result.prior_belief_masses() = Fusion::average_prior(opinions);
  }

  return result;
}

template <typename OpinionT>
OpinionT Fusion::cumulative_fusion_operator(const std::vector<OpinionT>& opinions,
                                            std::vector<typename OpinionT::FLOAT_t> uncertainties,
                                            typename OpinionT::FLOAT_t uncertainty_product)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  constexpr std::size_t N = OpinionT::SIZE;
  using FloatT = typename OpinionT::FLOAT_t;
  std::size_t n_elements = opinions.size();
  OpinionT result;

  // calculate the nominator, during calculation the belief_mass entries of the result are invalid
  for (std::size_t mass_idx{ 0 }; mass_idx < N; ++mass_idx)
  {
    for (std::size_t opinion_idx{ 0 }; opinion_idx < n_elements; ++opinion_idx)
    {
      result.belief_masses()[mass_idx] +=
          opinions[opinion_idx].belief_masses()[mass_idx] * uncertainty_product / uncertainties[opinion_idx];
    }
  }

  // sum of uncertainty products, where each product is omitting one specific uncertainty
  FloatT denom =
      std::accumulate(uncertainties.begin(), uncertainties.end(), 0., [uncertainty_product](FloatT sum, FloatT val) {
        return sum + uncertainty_product / val;
      });
  denom -= (n_elements - 1) * uncertainty_product;

  // normalize belief_masses using the precalculated denominator
  std::transform(result.belief_masses().begin(),
                 result.belief_masses().end(),
                 result.belief_masses().begin(),
                 [denom](FloatT val) { return val / denom; });

  return result;
}

template <typename OpinionT>
OpinionT Fusion::belief_constraint_fusion_operator(const std::vector<OpinionT>& opinions,
                                                   std::vector<typename OpinionT::FLOAT_t> uncertainties,
                                                   typename OpinionT::FLOAT_t uncertainty_product)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  OpinionT result;

  // since the belief constrained fusion is commutative, simply apply belief_constraint fusion sequentially
  for (auto const& opinion : opinions)
  {
    result.bc_fuse_(opinion);
  }

  return result;
}

template <typename OpinionT>
OpinionT Fusion::average_fusion_operator(const std::vector<OpinionT>& opinions,
                                         std::vector<typename OpinionT::FLOAT_t> uncertainties,
                                         typename OpinionT::FLOAT_t uncertainty_product)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  constexpr std::size_t N = OpinionT::SIZE;
  using FloatT = typename OpinionT::FLOAT_t;
  std::size_t n_elements = opinions.size();
  OpinionT result;

  // calculate the nominator, during calculation the belief_mass entries of the result are invalid
  for (std::size_t mass_idx{ 0 }; mass_idx < N; ++mass_idx)
  {
    for (std::size_t opinion_idx{ 0 }; opinion_idx < n_elements; ++opinion_idx)
    {
      result.belief_masses()[mass_idx] +=
          opinions[opinion_idx].belief_masses()[mass_idx] * uncertainty_product / uncertainties[opinion_idx];
    }
  }

  // sum of uncertainty products, where each product is omitting one specific uncertainty
  FloatT denom =
      std::accumulate(uncertainties.begin(), uncertainties.end(), 0., [uncertainty_product](FloatT sum, FloatT val) {
        return sum + uncertainty_product / val;
      });

  // normalize belief_masses using the precalculated denominator
  std::transform(result.belief_masses().begin(),
                 result.belief_masses().end(),
                 result.belief_masses().begin(),
                 [denom](FloatT val) { return val / denom; });

  return result;
}

}  // namespace subjective_logic::multisource
