#pragma once

#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/types/dirichlet_distribution.hpp"

namespace subjective_logic
{

template <std::size_t N, typename FloatT>
template <typename DirFloatT>
constexpr Opinion<N, FloatT>::Opinion(DirichletDistribution<N, DirFloatT> dirichlet)
  requires std::is_convertible_v<DirFloatT, FloatT>
  : Opinion{ dirichlet.evidences() / (dirichlet.evidences().sum() + N), dirichlet.priors() }
{
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>::operator DirichletDistribution<N, FloatT>() const
{
  return DirichletDistribution<N, FloatT>(*this);
}

template <std::size_t N, typename FloatT>
template <typename DirFloatT>
constexpr OpinionNoBase<N, FloatT>::OpinionNoBase(DirichletDistribution<N, DirFloatT> dirichlet)
  requires std::is_convertible_v<DirFloatT, FloatT>
  : OpinionNoBase{ dirichlet.evidences() / (dirichlet.evidences().sum() + N) }
{
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>::operator DirichletDistribution<N, FloatT>() const
{
  return DirichletDistribution<N, FloatT>(*this);
}

template <std::size_t N, typename FloatT>
template <typename OpFloatT>
constexpr DirichletDistribution<N, FloatT>::DirichletDistribution(OpinionNoBase<N, OpFloatT> opinion)
  requires std::is_convertible_v<OpFloatT, FloatT>
  : DirichletDistribution{ N * opinion.belief_masses() / opinion.uncertainty(), opinion.NeutralBeliefDistr() }
{
}

template <std::size_t N, typename FloatT>
template <typename OpFloatT>
constexpr DirichletDistribution<N, FloatT>::DirichletDistribution(Opinion<N, OpFloatT> opinion)
  requires std::is_convertible_v<OpFloatT, FloatT>
  : DirichletDistribution{ N * opinion.belief_masses() / opinion.uncertainty(), opinion.prior_belief_masses() }
{
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>::operator Opinion<N, FloatT>() const
{
  return Opinion<N, FloatT>(*this);
}

template <std::size_t N, typename FloatT>
constexpr DirichletDistribution<N, FloatT>::operator OpinionNoBase<N, FloatT>() const
{
  return OpinionNoBase<N, FloatT>(*this);
}
}  // namespace subjective_logic
