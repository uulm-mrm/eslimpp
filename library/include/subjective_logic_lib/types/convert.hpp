#pragma once

#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/types/dirichlet_distribution.hpp"

namespace subjective_logic
{

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>::operator DirichletDistribution<N, FloatT>() const
{
  DirichletDistribution<N, FloatT> dist(N * this->belief_masses() / this->uncertainty(), this->prior_belief_masses());
  return dist;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>::operator DirichletDistribution<N, FloatT>() const
{
  DirichletDistribution<N, FloatT> dist(N * this->belief_masses() / this->uncertainty(), this->NeutralBeliefDistr());
  return dist;
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>::operator Opinion<N, FloatT>() const
{
  FloatT denom = this->evidence_.sum() + N;
  Opinion<N, FloatT> opinion(this->evidence_ / denom, this->prior_);
  return opinion;
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>::operator OpinionNoBase<N, FloatT>() const
{
  FloatT denom = this->evidence_.sum() + N;
  OpinionNoBase<N, FloatT> opinion(this->evidence_ / denom);
  return opinion;
}
}  // namespace subjective_logic
