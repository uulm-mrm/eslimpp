#pragma once

#include <cstring>

#include <array>
#include <cassert>
#include <cmath>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/types/cuda_compatible_array.hpp"

// some papers are used as reference:
// [2] Scheible, Alexander et al.: Track Classification for Random Finite Set Based Multi-Sensor Multi-Object Tracking.
//     DOI: 10.1109/SDF-MFI59545.2023.10361438.
// [3] Lance Kaplan, et al.: Partial observable update for subjective logic and its application for trust estimation,
//     Information Fusion, Volume 26, 2015, Pages 66-83, ISSN 1566-2535, https://doi.org/10.1016/j.inffus.2015.01.005.

namespace subjective_logic
{

// forward declaration to allow declaration of conversion functions
template <std::size_t N, typename FloatT>
class Opinion;

template <std::size_t N, typename FloatT = float>
class DirichletDistribution
{
public:
  using FLOAT_t = FloatT;
  static constexpr std::size_t SIZE = N;
  using WeightType = Array<N, FloatT>;

  /**
   * @brief creates a vacuous distribution, meaning that all evidence is 0 and the prior is neutral
   */
  CUDA_AVAIL
  constexpr DirichletDistribution();

  /**
   * @brief creates an opinion using the given alpha weights, validity is not checked
   *        a neutral prior is assumed to allow obtaining evidence from alpha values
   */
  CUDA_AVAIL
  explicit constexpr DirichletDistribution(WeightType alphas);

  /**
   * @brief creates an opinion using given evidences and prior weights, validity is not checked
   */
  CUDA_AVAIL
  explicit constexpr DirichletDistribution(WeightType evidences, WeightType priors);

  /**
   * @brief allows to use the ctor with the correct number of arguments instead of requiring an initializer list
   *        when init a dirichlet distr. like this, the prior is set to a neutral distr
   *        since defining multiple parameter packs is possible, but the resulting constructors would become quite
   *        confusing
   */
  template <typename... VALUES>
  CUDA_AVAIL constexpr explicit DirichletDistribution(VALUES... values)
    requires(is_arithmetic_list<VALUES...> and sizeof...(VALUES) == N);

  /**
   * @brief default copy ctor
   * @param other
   */
  CUDA_AVAIL
  constexpr DirichletDistribution(const DirichletDistribution& other) = default;

  /**
   * @brief default move ctor
   * @param other
   */
  CUDA_AVAIL
  constexpr DirichletDistribution(DirichletDistribution&& other) = default;

  /**
   * @brief default copy assignment
   * @param other
   */
  CUDA_AVAIL
  constexpr DirichletDistribution& operator=(const DirichletDistribution& other) = default;

  /**
   * @brief default move assignment
   * @param other
   */
  CUDA_AVAIL
  constexpr DirichletDistribution& operator=(DirichletDistribution&& other) = default;

  /**
   * @brief default dtor (not virtual!!!!, see description above)
   */
  CUDA_AVAIL
  ~DirichletDistribution() = default;

  /**
   * @brief allows to construct a Dirichlet distribution with evidences only, ctor only allows alpha values
   *        for this, a neutral prior is assumed
   */
  static constexpr DirichletDistribution from_evidences(WeightType evidences);

  /**
   * @brief access to the evidence vector
   * @return reference to evidence vector
   */
  CUDA_AVAIL
  constexpr WeightType& evidences();
  /**
   * @brief const access to the evidence vector
   * @return const reference to evidence vector
   */
  CUDA_AVAIL
  constexpr const WeightType& evidences() const;

  /**
   * @brief access to the prior vector
   * @return reference to the prior vector
   */
  CUDA_AVAIL
  constexpr WeightType& priors();
  /**
   * @brief const access to the prior vector
   * @return const reference to the prior vector
   */
  CUDA_AVAIL
  constexpr const WeightType& priors() const;

  /**
   * @brief allows convenient access to the alpha parameters defined automatically combining evidence and prior
   * @return vector containing the alpha values of the dirichlet distributions
   */
  CUDA_AVAIL
  constexpr WeightType alphas() const;

  /**
   *  @brief convert to subjective logic opinion while differentiating between evidence and prior
   *  @return subjective logic opinion with prior
   */
  CUDA_AVAIL
  constexpr explicit operator Opinion<N, FloatT>() const;

  /**
   *  @brief convert to subjective logic opinion_no_base neglecting the prior information
   *  @return subjective logic opinion_no_base without prior
   */
  CUDA_AVAIL
  constexpr explicit operator OpinionNoBase<N, FloatT>() const;

  /**
   * @brief evaluates the dirichlet distribution PDF for a given a distribution over the N-1 simplex
   * @returns the scalar value of the dirichlet distribution
   */
  CUDA_AVAIL
  constexpr FloatT evaluate(WeightType distr) const;

  /**
   * @brief evaluates the dirichlet distribution PDF for a given probability in the binomial case
   * @returns the scalar value of the dirichlet distribution
   */
  CUDA_AVAIL
  constexpr FloatT evaluate(FloatT p) const
    requires is_binomial<N>;

  /**
   * @brief calculates the mean distribution of the dirichlet distribution
   * @return mean
   */
  CUDA_AVAIL
  constexpr WeightType mean() const;

  /**
   * @brief calculates the mean distribution of the dirichlet distribution in the binomial case
   * @return mean
   */
  CUDA_AVAIL
  constexpr FloatT mean_binomial() const
    requires is_binomial<N>;

  /**
   * @brief calculates the variance of the dirichlet distribution elementwise, i.e., no cross correlation is considered
   * @return variance
   */
  CUDA_AVAIL
  constexpr WeightType variance() const;

  /**
   * @brief implements the dirichlet distribution update proposed by [2] inplace
   *        It is meant for updates, when the outcome of an experiment is
   *        not clearly observed as one possible state (e.g. detections from a deep leaning detector)
   */
  CUDA_AVAIL
  constexpr DirichletDistribution& moment_matching_update_(WeightType probabilities);

  /**
   * @brief implementation of the above operator including a copy
   */
  CUDA_AVAIL
  constexpr DirichletDistribution moment_matching_update(WeightType probabilities) const;

protected:
  WeightType evidence_;
  WeightType prior_;
};

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>::DirichletDistribution()
  : DirichletDistribution(WeightType{ 0 }, WeightType{ 1 / static_cast<FloatT>(N) })
{
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>::DirichletDistribution(WeightType alphas)
  : DirichletDistribution(alphas, WeightType{ 1 / static_cast<FloatT>(N) })
{
  evidence_ -= N * prior_;
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>::DirichletDistribution(WeightType evidences, WeightType priors)
  : evidence_{ evidences }, prior_{ priors }
{
}

template <std::size_t N, typename FloatT>
template <typename... VALUES>
constexpr DirichletDistribution<N, FloatT>::DirichletDistribution(VALUES... values)
  requires(is_arithmetic_list<VALUES...> and sizeof...(VALUES) == N)
  : DirichletDistribution(WeightType{ static_cast<FloatT>(values)... }, WeightType{ 1 / static_cast<FloatT>(N) })
{
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT> DirichletDistribution<N, FloatT>::from_evidences(WeightType evidences)
{
  return DirichletDistribution{ evidences, WeightType{ 1 / static_cast<FloatT>(N) } };
}

template <std::size_t N, typename FloatT>
constexpr typename DirichletDistribution<N, FloatT>::WeightType& DirichletDistribution<N, FloatT>::evidences()
{
  return evidence_;
}

template <std::size_t N, typename FloatT>
constexpr const typename DirichletDistribution<N, FloatT>::WeightType&
DirichletDistribution<N, FloatT>::evidences() const
{
  return evidence_;
}

template <std::size_t N, typename FloatT>
constexpr typename DirichletDistribution<N, FloatT>::WeightType& DirichletDistribution<N, FloatT>::priors()
{
  return prior_;
}

template <std::size_t N, typename FloatT>
constexpr const typename DirichletDistribution<N, FloatT>::WeightType& DirichletDistribution<N, FloatT>::priors() const
{
  return prior_;
}

template <std::size_t N, typename FloatT>
constexpr typename DirichletDistribution<N, FloatT>::WeightType DirichletDistribution<N, FloatT>::alphas() const
{
  return evidence_ + N * prior_;
}

template <std::size_t N, typename FloatT>
constexpr FloatT DirichletDistribution<N, FloatT>::evaluate(WeightType distr) const
{
  auto alphas = this->alphas();

  FloatT value = std::tgamma(alphas.sum());
  constexpr_for<0, N, 1>([alphas, distr, &value](std::size_t idx) {
    auto distr_value = distr[idx];
    auto alpha_value = alphas[idx];
    // the absolute value of the samples distr must be strictly greater than 0, if alpha is smaller than 1,
    // however, to avoid throwing an error or further handling, 0 is returned for practical reasons.
    if (std::abs(distr_value) < EPS_v<FloatT> and alpha_value < 1)
    {
      value = 0;
      return;
    }
    value *= std::pow(distr_value, (alpha_value - 1)) / std::tgamma(alpha_value);
  });

  return value;
}

template <std::size_t N, typename FloatT>
constexpr FloatT DirichletDistribution<N, FloatT>::evaluate(FloatT p) const
  requires is_binomial<N>
{
  return this->evaluate(WeightType{ p, 1 - p });
}

template <std::size_t N, typename FloatT>
constexpr typename DirichletDistribution<N, FloatT>::WeightType DirichletDistribution<N, FloatT>::mean() const
{
  auto alphas = this->alphas();
  auto sum = alphas.sum();
  return alphas / sum;
}

template <std::size_t N, typename FloatT>
constexpr FloatT DirichletDistribution<N, FloatT>::mean_binomial() const
  requires is_binomial<N>
{
  auto alphas = this->alphas();
  auto sum = alphas.sum();
  return alphas[0] / sum;
}

template <std::size_t N, typename FloatT>
constexpr typename DirichletDistribution<N, FloatT>::WeightType DirichletDistribution<N, FloatT>::variance() const
{
  auto alphas = this->alphas();
  auto sum = alphas.sum();
  auto alpha_tilde = alphas / sum;
  return alpha_tilde * (static_cast<FloatT>(1.0) - alpha_tilde) / (sum + static_cast<FloatT>(1.0));
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>&
DirichletDistribution<N, FloatT>::moment_matching_update_(WeightType probabilities)
{
  WeightType alphas = this->alphas();
  FloatT S = alphas.sum();

  WeightType moments;
  WeightType variances;
  WeightType nom;
  WeightType denom;
  constexpr_for<0, N, 1>([S, alphas, probabilities, &moments, &variances, &nom, &denom](std::size_t idx) {
    moments[idx] = (alphas[idx] + probabilities[idx]) / (static_cast<FloatT>(1.0) + S);
    variances[idx] = (1 + alphas[idx]) * (alphas[idx] + 2 * probabilities[idx]) /
                     ((static_cast<FloatT>(1.0) + S) * (static_cast<FloatT>(2.0) + S));
    // there is an error in the publicly available version of [2], the square accidentally jumped to the outside of the
    // parenthesis following [3] here, the moment inside the parenthesis must be squared
    FloatT tmp = moments[idx] * (static_cast<FloatT>(1.0) - moments[idx]);
    nom[idx] = (moments[idx] - variances[idx]) * tmp;
    denom[idx] = (variances[idx] - moments[idx] * moments[idx]) * tmp;
  });
  FloatT factor = nom.sum() / denom.sum();

  // [3] shows an additional term which did not come into play in the experiments, hence, it is commented out for now.
  // moments should probably sum up to one with the formula above
  // moments /= moments.sum();

  // [3] shows an additional term which did not come into play in the experiments, hence, it is commented out for now.
  // FloatT factor_limit = max<0,N>([&, this](std::size_t idx) -> FloatT { return N * prior_[idx]/moments[idx]; });
  // if (factor < factor_limit) {
  //   factor = factor_limit;
  // }

  WeightType new_alphas = moments * factor;

  evidence_ = new_alphas - N * this->prior_;
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>
DirichletDistribution<N, FloatT>::moment_matching_update(WeightType probabilities) const
{
  return DirichletDistribution{ *this }.moment_matching_update_(probabilities);
}

}  // namespace subjective_logic

#include "subjective_logic_lib/types/convert.hpp"
