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
#include "subjective_logic_lib/types/fusion_types.hpp"
#include "subjective_logic_lib/types/cuda_compatible_array.hpp"
#include "subjective_logic_lib/opinions/opinion.hpp"

// used in later tests also for protected member functions

namespace subjective_logic::multisource
{

struct Fusion
{
  template <typename OpinionT>
  static inline OpinionT fuse_opinions(FusionType fusion_type, std::vector<OpinionT> opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  template <typename OpinionT>
  static inline OpinionT fuse_opinions(FusionType fusion_type, std::initializer_list<OpinionT>& inputs)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  template <typename... Opinions>
  static inline typename FirstType<Opinions...>::type fuse_opinions(FusionType fusion_type, Opinions... opinions)
    requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>;

  // template <std::size_t N, typename OpinionT>
  // CUDA_AVAIL static inline OpinionT fuse_opinions(FusionType fusion_type, const Array<N,OpinionT>& inputs);

protected:
  // FusionOperator is used internally, to type the functions for the specific operator implementation
  template <typename OpinionT>
  using FusionOperator = std::function<
      OpinionT(const std::vector<OpinionT>&, std::vector<typename OpinionT::FLOAT_t>, typename OpinionT::FLOAT_t)>;

  /**
   * preprocessing steps of all multi-source fusion operators are combined in this function
   * in case that some opinions are dogmatic, the result is almost always just a mean of all dogmatic opinions,
   * thus, in this case, the result is precalculated a returned by this function
   * @param opinions
   * @return
   */
  template <typename OpinionT>
  static inline std::tuple<std::vector<typename OpinionT::FLOAT_t>, std::optional<OpinionT>>
  preprocess_opinions(const std::vector<OpinionT>& opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * the prior is not handled by any multi-source fusion model, instead it gets averaged over all available opinions
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
   * @param opinions
   * @param fusion_operator
   * @return
   */
  template <typename OpinionT>
  static inline OpinionT fuse_opinions_(const std::vector<OpinionT>& opinions, FusionOperator<OpinionT> fusion_operator)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * includes all operator-specific calculations and is used together with fuse_opinions
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
   * includes all operator-specific calculations and is used together with fuse_opinions
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
   * includes all operator-specific calculations and is used together with fuse_opinions
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

  // in order to allow cuda available generic lambda functions, the function access must be public
  // necessity might vanish, when not using generic lambdas
  // access for test function may be provided using a Test friend class
public:
  // template <std::size_t N, typename OpinionT>
  // using FusionOperatorArr = CUDA_AVAIL std::function<
  //     OpinionT(const Array<N,OpinionT>&, Array<N, typename OpinionT::FLOAT_t>, typename OpinionT::FLOAT_t)>;

  template <std::size_t N, typename OpinionT>
  CUDA_AVAIL static inline OpinionT fuse_opinions(FusionType fusion_type, const Array<N, OpinionT>& opinions)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * the prior is not handled by any multi-source fusion model, instead it gets averaged over all available opinions
   * this function implements a cuda ready implementation with a fixed number of opinions
   * @tparam OpinionT
   * @param opinions
   * @return
   */
  template <std::size_t N, typename OpinionT>
  CUDA_AVAIL static inline typename OpinionT::BeliefType average_prior(const Array<N, OpinionT>& opinions)
    requires is_opinion<OpinionT>;

  /**
   * preprocessing steps of all multi-source fusion operators are combined in this function
   * in case that some opinions are dogmatic, the result is almost always just a mean of all dogmatic opinions,
   * thus, in this case, the result is precalculated a returned by this function.
   * this function implements a cuda ready implementation with a fixed number of opinions
   * @param opinions
   * @return
   */
  template <std::size_t N, typename OpinionT>
  CUDA_AVAIL static inline bool preprocess_opinions(const Array<N, OpinionT>& opinions,
                                                    Array<N, typename OpinionT::FLOAT_t>& uncertainties)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * includes all operator-specific calculations and is used together with fuse_opinions
   * this function implements a cuda ready implementation with a fixed number of opinions
   * @tparam OpinionT
   * @param opinions
   * @param uncertainties
   * @param uncertainty_product
   * @return
   */
  template <std::size_t N, typename OpinionT>
  CUDA_AVAIL static inline OpinionT cumulative_fusion_operator(const Array<N, OpinionT>& opinions,
                                                               Array<N, typename OpinionT::FLOAT_t> uncertainties,
                                                               typename OpinionT::FLOAT_t uncertainty_product)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;

  /**
   * fuse all opinions using a given fusion operator
   * this functions handles everything except the actual fusion operation
   * the given fusion operator is provided with precalculated uncertainties and the uncertainty product
   * further, using the preprocess_opinions function, dogmatic opinions are handled.
   * @param opinions
   * @param fusion_operator
   * @return
   */
  template <std::size_t N, typename OpinionT, typename Func>
  CUDA_AVAIL static inline OpinionT fuse_opinions_(const Array<N, OpinionT>& opinions, Func&& fusion_operator)
    requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>;
};

template <typename OpinionT>
inline OpinionT Fusion::fuse_opinions(FusionType fusion_type, std::initializer_list<OpinionT>& inputs)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  return fuse_opinions(fusion_type, std::vector<OpinionT>{ inputs });
}

template <typename... Opinions>
inline typename FirstType<Opinions...>::type Fusion::fuse_opinions(FusionType fusion_type, Opinions... opinions)
  requires is_opinion_no_base_list<Opinions...> or is_opinion_list<Opinions...>
{
  using OutType = typename FirstType<Opinions...>::type;
  return OutType{ fuse_opinions(fusion_type, std::vector<OutType>{ opinions... }) };
}

// template <std::size_t N, typename OpinionT>
// CUDA_AVAIL inline OpinionT Fusion::fuse_opinions(FusionType fusion_type, const Array<N,OpinionT>& inputs) {
//
//   return inputs.at(N);
// }

template <typename OpinionT>
inline OpinionT Fusion::fuse_opinions(FusionType fusion_type, std::vector<OpinionT> opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  switch (fusion_type)
  {
    case FusionType::CUMULATIVE:
    {
      return fuse_opinions_<OpinionT>(opinions, &Fusion::cumulative_fusion_operator<OpinionT>);
    }
    case FusionType::BELIEF_CONSTRAINT:
    {
      return fuse_opinions_<OpinionT>(opinions, &Fusion::belief_constraint_fusion_operator<OpinionT>);
    }
    case FusionType::AVERAGE:
    {
      return fuse_opinions_<OpinionT>(opinions, &Fusion::average_fusion_operator<OpinionT>);
    }
    default:
    {
      throw std::logic_error{ "MultiSource fusion is not yet implemented for: " +
                              std::to_string(static_cast<int>(fusion_type)) };
    }
  }
}

template <std::size_t N, typename OpinionT>
inline OpinionT Fusion::fuse_opinions(FusionType fusion_type, const Array<N, OpinionT>& opinions)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  switch (fusion_type)
  {
    case FusionType::CUMULATIVE:
    {
      return fuse_opinions_<N, OpinionT>(opinions, Fusion::cumulative_fusion_operator<N, OpinionT>);
    }
    // case FusionType::BELIEF_CONSTRAINT:
    // {
    //   return fuse_opinions_(opinions, Fusion::belief_constraint_fusion_operator<OpinionT>);
    // }
    // case FusionType::AVERAGE:
    // {
    //   return fuse_opinions_(opinions, Fusion::average_fusion_operator<OpinionT>);
    // }
    default:
    {
      // should not be used, it's more like spamming the console to tell the user somethings off
      printf("fusion func not yet implemented");
      return OpinionT{};
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

template <std::size_t N, typename OpinionT>
CUDA_AVAIL inline typename OpinionT::BeliefType Fusion::average_prior(const Array<N, OpinionT>& opinions)
  requires is_opinion<OpinionT>
{
  using ReturnType = typename OpinionT::BeliefType;
  ReturnType prior{ 0 };

  constexpr_for<0, N>([&](std::size_t opin_idx) {
    constexpr_for<0, OpinionT::SIZE>(
        [&](std::size_t idx) { prior[idx] += opinions[opin_idx].prior_belief_masses()[idx]; });
  });

  constexpr_for<0, OpinionT::SIZE>([&](std::size_t idx) { prior[idx] /= opinions.size(); });

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

template <std::size_t N, typename OpinionT>
inline bool Fusion::preprocess_opinions(const Array<N, OpinionT>& opinions,
                                        Array<N, typename OpinionT::FLOAT_t>& uncertainties)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  using InputArray = const Array<N, OpinionT>;

  if constexpr (N == 1)
  {
    return false;
  }

  constexpr_for<0, N>([&](std::size_t idx, InputArray& opins) { uncertainties[idx] = opins[idx].uncertainty(); },
                      opinions);

  // todo(wodtko) check for dogmatic for cuda impl here
  return true;
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

template <std::size_t N, typename OpinionT, typename Func>
inline OpinionT Fusion::fuse_opinions_(const Array<N, OpinionT>& opinions, Func&& fusion_operator)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  using FloatT = typename OpinionT::FLOAT_t;
  Array<N, FloatT> uncertainties;
  // auto [uncertainties, pre_result] = Fusion::preprocess_opinions(opinions);
  if (not Fusion::preprocess_opinions(opinions, uncertainties))
  {
    return opinions[0];
  }

  // the product of all uncertainties can later be used to calculate specific products
  // by dividing the respective uncertainty which is omitted
  FloatT uncert_prod{ 1. };
  constexpr_for<0, N>([&](std::size_t uncert_idx) { uncert_prod *= uncertainties[uncert_idx]; });

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

template <std::size_t N, typename OpinionT>
inline OpinionT Fusion::cumulative_fusion_operator(const Array<N, OpinionT>& opinions,
                                                   Array<N, typename OpinionT::FLOAT_t> uncertainties,
                                                   typename OpinionT::FLOAT_t uncertainty_product)
  requires is_opinion<OpinionT> or is_opinion_no_base<OpinionT>
{
  using FloatT = typename OpinionT::FLOAT_t;
  std::size_t n_elements = opinions.size();
  OpinionT result;

  // calculate the nominator, during calculation the belief_mass entries of the result are invalid
  constexpr_for<0, OpinionT::SIZE>([&](std::size_t mass_idx) {
    constexpr_for<0, N>([&](std::size_t opin_idx) {
      result.belief_masses()[mass_idx] +=
          opinions[opin_idx].belief_masses()[mass_idx] * uncertainty_product / uncertainties[opin_idx];
    });
  });

  FloatT denom{ 0. };
  constexpr_for<0, N>([&](std::size_t uncert_idx) { denom += uncertainty_product / uncertainties[uncert_idx]; });

  denom -= (n_elements - 1) * uncertainty_product;

  constexpr_for<0, OpinionT::SIZE>([&, denom](std::size_t mass_idx) { result.belief_masses()[mass_idx] /= denom; });

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
