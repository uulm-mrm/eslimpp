#pragma once

// the reader is invited to refer to the following book as reference for the implementations within this file:
// [1]  Subjective Logic - A Formalism for Reasoning Under Uncertainty,
//      Audun Jøsang, 2016, https://doi.org/10.1007/978-3-319-42337-1
// [2]  Uncertainty Characteristics of Subjective Opinions
//      Audun Josang; Jin-Hee Cho; Feng Chen, https://doi.org/10.23919/ICIF.2018.8455454

#include <array>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/types/cuda_compatible_array.hpp"

namespace subjective_logic
{

// forward declaration to allow the declaration of convert functions
template <std::size_t N, typename FloatT>
class DirichletDistribution;

/**
 * @brief this class is meant to be used in large arrays and,
 * thus, size and fast access are sometime more important than an easy and intuitive use.
 * e.g. a v-table requires too much space, hence it should be avoided
 * when having 2 float values (binomial Opinion), the v-table doubles the required memory space
 * IMPORTAT!!!!!
 * this means, that there must not be any virtual function (especially ctor and dtor require special care)
 * it is not recommended to inherit from this class in any way, rather have it as member.
 * however, if required, only inherit from this class in a protected manner to avoid unintended base class casts.
 * child classes may not be cast to the base class using pointer interfaces!!!
 *
 * @tparam N dimension of the subjective logic opinion (2 = Binomial, >2 = multinomial)
 * @tparam FloatT float type which allows a compact but less accurate storage with float,
 *                or a less compact and more precise storage with double
 */
template <std::size_t N = 2, typename FloatT = float>
class OpinionNoBase
{
public:
  using BeliefType = Array<N, FloatT>;

  // helper to have accessible types/values when used from outside;
  using FLOAT_t = FloatT;
  static constexpr std::size_t SIZE = N;

  /**
   * @brief creates a vacuous opinion, meaning that all belief masses are 0 and the uncertainty is 1
   */
  constexpr OpinionNoBase() = default;

  /**
   * @brief creates an opinion using the given belief_mass distribution, validity is not checked
   * @param belief_masses
   */
  CUDA_AVAIL
  explicit constexpr OpinionNoBase(BeliefType belief_masses);

  /**
   * @brief allows to use the ctor with the correct number of arguments instead of requiring an initializer list
   */
  template <typename... VALUES>
  CUDA_AVAIL constexpr explicit OpinionNoBase(VALUES... values)
    requires(is_arithmetic_list<VALUES...> and sizeof...(VALUES) == N);

  /**
   * @brief default copy ctor
   * @param other
   */
  constexpr OpinionNoBase(const OpinionNoBase& other) = default;

  /**
   * @brief default move ctor
   * @param other
   */
  constexpr OpinionNoBase(OpinionNoBase&& other) = default;

  /**
   * @brief default copy assignment
   * @param other
   */
  constexpr OpinionNoBase& operator=(const OpinionNoBase& other) = default;

  /**
   * @brief default move assignment
   * @param other
   */
  constexpr OpinionNoBase& operator=(OpinionNoBase&& other) = default;

  /**
   * @brief default dtor (not virtual!!!!, see description above)
   */
  ~OpinionNoBase() = default;

  /**
   * @brief checks for the sum of belief_masses to be 1 up to an error of EPS<FloatT>
   */
  constexpr bool is_valid() const;

  /**
   * @brief allows a convenient access to the belief_distribution since it is inherited its access must be explicitly
   * defined
   * @return
   */
  CUDA_AVAIL
  constexpr BeliefType& belief_masses();
  /**
   * @brief allows a convenient access to the belief_distribution since it is inherited its access must be explicitly
   * defined
   * @return
   */
  CUDA_AVAIL
  constexpr const BeliefType& belief_masses() const;

  /**
   * @brief allows a convenient access to a belief_distribution entry
   * defined
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT& belief_mass(std::size_t idx);

  /**
   * @brief allows a convenient access to a belief_distribution entry
   * defined
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT belief_mass(std::size_t idx) const;

  /**
   * @brief calculates the uncertainty of this opinion, should be stored by the user if used more often
   * @return the uncertainty of this opinion in a subjective logic sense
   */
  CUDA_AVAIL
  [[nodiscard]] constexpr FloatT uncertainty() const;
  /**
   * @brief calculates the Dirichlet evidence vector given W equals the opinion's dimension
   * @return evidence vector
   */
  CUDA_AVAIL
  [[nodiscard]] constexpr Array<N, FloatT> evidence() const;

  /**
   * @brief convenient and read/writeable access to the first belief_masses entry for binomial Opinions
   * @return belief of a binomial opinion
   */
  CUDA_AVAIL
  constexpr FloatT& belief()
    requires is_binomial<N>;

  /**
   * @brief convenient and readable access to the first belief_masses entry for binomial Opinions
   * @return belief of a binomial opinion
   */
  CUDA_AVAIL
  constexpr const FloatT& belief() const
    requires is_binomial<N>;

  /**
   * @brief convenient and read/writeable access to the second belief_masses entry for binomial Opinions
   * @return disbelief of a binomial opinion
   */
  CUDA_AVAIL
  constexpr inline FloatT& disbelief()
    requires is_binomial<N>;

  /**
   * @brief convenient and readable access to the second belief_masses entry for binomial Opinions
   * @return disbelief of a binomial opinion
   */
  CUDA_AVAIL
  constexpr inline const FloatT& disbelief() const
    requires is_binomial<N>;

  /**
   * @brief calculates the complement of an opinion (switching belief and disbelief)
   * @return complement of a binomial opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase complement() const
    requires is_binomial<N>;

  /**
   * @brief linearly interpolates between to opinions using the given factor
   *        factor of 0 return this, factor of 1 returns other.
   *        While there is no direct interpretation of this operation,
   *        it can be used to form transitions between opinions.
   * @param other
   * @param interp_fac
   * @return interpolated opinion using a copy
   */
  CUDA_AVAIL
  constexpr OpinionNoBase interpolate(OpinionNoBase other, FloatT interp_fac) const;

  /**
   * @brief a neutral distribution assigns the an equal amount of evidence to each belief_mass entry
   *        the resulting belief distribution sums up to 1, i.e., if interpreted as an Opinion, the uncertainty would be
   * 0.
   * @return a neutral opinion
   */
  CUDA_AVAIL
  static constexpr BeliefType NeutralBeliefDistr();

  /**
   * @brief a vacuous distribution assigns the no evidence no any hypotheses
   *        the resulting uncertainty is 1.
   *        this function returns a belief distr similar one created by the default ctor, however the meaning/context
   * might be more clear to the reader 0.
   * @return a vacuous opinion
   */
  CUDA_AVAIL
  static constexpr BeliefType VacuousBeliefDistr();

  /**
   * @brief creates an OpinionNoBase with a neutral belief distribution
   * @return
   */
  CUDA_AVAIL
  static constexpr OpinionNoBase NeutralBeliefOpinion();

  /**
   * @brief creates an OpinionNoBase with a vacuous belief distribution
   * @return
   */
  CUDA_AVAIL
  static constexpr OpinionNoBase VacuousBeliefOpinion();

  /**
   * @brief calculates the dissonance of an opinion as proposed in [2]
   * @return the dissonance
   */
  CUDA_AVAIL
  constexpr FloatT dissonance() const;

  /**
   * @brief calculates the probability with respect to the sum of belief mass available
   *        e.g., the probability of the respective uncertainty minimised Opinion.
   *        the output is scalar to allow an easier use with binomial Opinions.
   * @return the probability of the variable being true, without considering the uncertainty
   */
  CUDA_AVAIL
  constexpr FloatT getProbability() const
    requires is_binomial<N>;

  /**
   * @brief calculates the probability with respect to the sum of belief mass available
   *        e.g., the probability of the respective uncertainty minimised Opinion.
   * @return the probability without considering the uncertainty
   */
  CUDA_AVAIL
  constexpr BeliefType getProbabilities() const;

  /**
   * @brief calculates the projected probability, for this, the base rate must be available
   *        the output is scalar to allow an easier use with binomial Opinions.
   * @return the projected probability
   */
  CUDA_AVAIL
  constexpr FloatT getBinomialProjection(FloatT base_rate = 0.5) const
    requires is_binomial<N>;

  /**
   * @brief calculates the projected probability, for this, the base rate must be available
   * @param base_rate
   * @return the projected probability
   */
  CUDA_AVAIL
  [[nodiscard]] constexpr BeliefType getProjection(BeliefType base_rate) const;

  /**
   * @brief calculates the uncertainty differential w.r.t. another opinion
   * @param other
   * @return the uncertainty differential
   */
  CUDA_AVAIL
  constexpr FloatT uncertainty_differential(OpinionNoBase other) const;

  /**
   * @brief calculates the degree of conflict w.r.t. another opinion, for this, both base rates must be available
   *        this function allows a scalar base rate for binomial opinions
   * @param other
   * @param base_rate base rate of this opinion
   * @param base_rate_other base rate of the other opinion
   * @return the degree of conflict
   */
  CUDA_AVAIL
  constexpr FloatT degree_of_conflict(OpinionNoBase other, FloatT base_rate, FloatT base_rate_other) const
    requires is_binomial<N>;

  /**
   * @brief calculates the degree of conflict w.r.t. another opinion, for this, both base rates must be available
   * @param other
   * @param base_rate base rate of this opinion
   * @param base_rate_other base rate of the other opinion
   * @return the degree of conflict
   */
  CUDA_AVAIL
  constexpr FloatT degree_of_conflict(OpinionNoBase other,
                                      BeliefType base_rate = NeutralBeliefDistr(),
                                      BeliefType base_rate_other = NeutralBeliefDistr()) const;

  /**
   * @brief calculates the degree of harmony w.r.t. another opinion, for this, both base rates must be available
   *        this function allows a scalar base rate for binomial opinions
   * @param other
   * @param base_rate base rate of this opinion
   * @param base_rate_other base rate of the other opinion
   * @return the degree of harmony
   */
  CUDA_AVAIL
  constexpr FloatT degree_of_harmony(OpinionNoBase other, FloatT base_rate, FloatT base_rate_other) const
    requires is_binomial<N>;

  /**
   * @brief calculates the degree of harmony w.r.t. another opinion, for this, both base rates must be available
   * @param other
   * @param base_rate base rate of this opinion
   * @param base_rate_other base rate of the other opinion
   * @return the degree of harmony
   */
  CUDA_AVAIL
  constexpr FloatT degree_of_harmony(OpinionNoBase other,
                                     BeliefType base_rate = NeutralBeliefDistr(),
                                     BeliefType base_rate_other = NeutralBeliefDistr()) const;

  /**
   * @brief applies the concept of trust revision of [1] inplace, this is only applicable for trust opinions (binomial)
   * @param degree_of_conflict
   * @param other
   * @return reference to this
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& revise_trust_(FloatT degree_of_conflict, OpinionNoBase other)
    requires is_binomial<N>;

  /**
   * @brief applies the concept of trust revision of [1] using a copy, this is only applicable for trust opinions
   * (binomial)
   * @param degree_of_conflict
   * @param other
   * @return revised copy of this
   */
  CUDA_AVAIL
  constexpr OpinionNoBase revise_trust(FloatT degree_of_conflict, OpinionNoBase other) const
    requires is_binomial<N>;

  /**
   * @brief applies the concept of trust revision of [1] inplace, this is only applicable for trust opinions (binomial)
   * @param revision_factor
   * @return reference to this
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& revise_trust_(FloatT revision_factor)
    requires is_binomial<N>;

  /**
   * @brief applies the concept of trust revision of [1] inplace, this is only applicable for trust opinions (binomial)
   * @param revision_factor
   * @return reference to this
   */
  CUDA_AVAIL
  constexpr OpinionNoBase revise_trust(FloatT revision_factor) const
    requires is_binomial<N>;

  /**
   * @brief applies the concept of multiplication of [1] inplace
   * @param other
   * @param base_this
   * @param base_other
   * @return multiplied opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& multiply_(OpinionNoBase other, FloatT base_this = 0.5, FloatT base_other = 0.5)
    requires is_binomial<N>;

  /**
   * @brief applies the concept of multiplication of [1] using a copy
   * @param other
   * @param base_this
   * @param base_other
   * @return multiplied opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase multiply(OpinionNoBase other, FloatT base_this = 0.5, FloatT base_other = 0.5) const
    requires is_binomial<N>;

  /**
   * @brief applies the concept of comultiplication of [1] inplace
   * @param other
   * @return comultiplied opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& comultiply_(OpinionNoBase other, FloatT base_this = 0.5, FloatT base_other = 0.5)
    requires is_binomial<N>;

  /**
   * @brief applies the concept of comultiplication of [1] using a copy
   * @param other
   * @return comultiplied opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase comultiply(OpinionNoBase other, FloatT base_this = 0.5, FloatT base_other = 0.5) const
    requires is_binomial<N>;

  /**
   * @brief applies the concept of cumulative belief fusion of [1] inplace
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& cum_fuse_(OpinionNoBase other);
  /**
   * @brief applies the concept of cumulative belief fusion of [1] using a copy
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase cum_fuse(OpinionNoBase other) const;
  /**
   * @brief applies the concept of cumulative unfusion of [1] inplace
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& cum_unfuse_(OpinionNoBase other);
  /**
   * @brief applies the concept of cumulative unfusion of [1] using a copy
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase cum_unfuse(OpinionNoBase other) const;

  /**
   * @brief calculates the harmony of two opinion, which is later used for the belief constrained fusion
   * @param other
   * @return the harmony (ca be interpreted as a function of the domain X, meaning that the harmony differs for each
   * hypothesis)
   */
  CUDA_AVAIL
  constexpr BeliefType harmony(OpinionNoBase other) const;
  /**
   * @brief calculates the conflict of two opinion, which is later used for the belief constrained fusion
   * @param other
   * @return the conflict
   */
  CUDA_AVAIL
  constexpr FloatT conflict(OpinionNoBase other) const;
  /**
   * @brief applies the concept of belief constrained fusion of [1] inplace
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& bc_fuse_(OpinionNoBase other);
  /**
   * @brief applies the concept of belief constrained fusion of [1] using a copy
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase bc_fuse(OpinionNoBase other) const;

  /**
   * @brief applies the concept of averaging belief fusion of [1] inplace
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& average_fuse_(OpinionNoBase other);
  /**
   * @brief applies the concept of averaging belief fusion of [1] using a copy
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase average_fuse(OpinionNoBase other) const;
  /**
   * @brief applies the concept of averaging belief unfusion of [1] inplace
   * @return the unfused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& average_unfuse_(OpinionNoBase other);
  /**
   * @brief applies the concept of averaging belief unfusion of [1] using a copy
   * @param other
   * @return the unfused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase average_unfuse(OpinionNoBase other) const;

  /**
   * @brief applies the concept of weighted belief fusion of [1] inplace
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& wb_fuse_(OpinionNoBase other);
  /**
   * @brief applies the concept of weighted belief fusion of [1] using a copy
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase wb_fuse(OpinionNoBase other) const;

  // I (wodtko) am not sure, if a use of this operator is useful without considering hyperopinions.
  // the way it is implemented it simply considers all opinions to be singletons
  /**
   * @brief applies the concept of consensus and compromise belief fusion of [1] inplace
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& cc_fuse_(OpinionNoBase other);
  /**
   * @brief applies the concept of consensus and compromise belief fusion of [1] using a copy
   * @param other
   * @return the fused opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase cc_fuse(OpinionNoBase other) const;

  /**
   * @brief applies the concept of moment matching inline
   * @param probabilities
   * @return the updated opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& moment_matching_update_(BeliefType probabilities);

  /**
   * @brief applies the concept of moment matching using a copy
   * @param probabilities
   * @return the updated opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase moment_matching_update(BeliefType probabilities) const;

  /**
   * @brief applies the concept of trust discounting of [1] inplace, when using a Opinion for discounting its base rate
   * must be available the projected probability of the other trust Opinion tells, how much information is kept. i.e.,
   * prop=1 means no discounting, prop=0 means the output has an uncertainty of 1.
   * @param other
   * @param base_rate
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& trust_discount_(OpinionNoBase<2, FloatT> other, FloatT base_rate = 0.5);
  /**
   * @brief applies the concept of trust discounting of [1] using a copy, when using a Opinion for discounting its base
   * rate must be available
   * @param other
   * @param base_rate
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase trust_discount(OpinionNoBase<2, FloatT> other, FloatT base_rate = 0.5) const;
  /**
   * @brief applies the concept of trust discounting of [1] inplace
   *        the probability is interpreted as the projected probability of a trust opinion
   * @param prop
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& trust_discount_(FloatT prop);
  /**
   * @brief applies the concept of trust discounting of [1] using a copy
   *        the probability is interpreted as the projected probability of a trust opinion
   * @param prop
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase trust_discount(FloatT prop) const;

  /**
   * @brief applies the concept of trust discounting of [1] inplace
   *        in addition to the trust discount a limit for the resulting uncertainty can be set.
   *        the resulting opinion must only be as uncertain as defined by this limit.
   *        IMPORANT: if the uncertainty is already higher than the limit before, this function results in a trust
   * discount similar to rate=1.0.
   * @param limit
   * @param other
   * @param base_rate
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& limited_trust_discount_(FloatT limit,
                                                   OpinionNoBase<2, FloatT> other,
                                                   FloatT base_rate = 0.5);
  /**
   * @brief applies the concept of trust discounting of [1] using a copy
   *        in addition to the trust discount a limit for the resulting uncertainty can be set.
   *        the resulting opinion must only be as uncertain as defined by this limit.
   *        IMPORANT: if the uncertainty is already higher than the limit before, this function results in a trust
   * discount similar to rate=1.0.
   * @param limit
   * @param other
   * @param base_rate
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase limited_trust_discount(FloatT limit,
                                                 OpinionNoBase<2, FloatT> other,
                                                 FloatT base_rate = 0.5) const;
  /**
   * @brief applies the concept of trust discounting of [1] inplace
   *        the probability is interpreted as the projected probability of a trust opinion
   *        in addition to the trust discount a limit for the resulting uncertainty can be set.
   *        the resulting opinion must only be as uncertain as defined by this limit.
   *        IMPORANT: if the uncertainty is already higher than the limit before, this function results in a trust
   * discount similar to rate=1.0.
   * @param limit
   * @param prop
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& limited_trust_discount_(FloatT limit, FloatT prop);
  /**
   * @brief applies the concept of trust discounting of [1] using a copy
   *        the probability is interpreted as the projected probability of a trust opinion
   *        in addition to the trust discount a limit for the resulting uncertainty can be set.
   *        the resulting opinion must only be as uncertain as defined by this limit.
   *        IMPORANT: if the uncertainty is already higher than the limit before, this function results in a trust
   * discount similar to rate=1.0.
   * @param limit
   * @param prop
   * @return the trust discounted opinion
   */
  CUDA_AVAIL
  constexpr OpinionNoBase limited_trust_discount(FloatT limit, FloatT prop) const;

  /**
   * @brief applies the concept of deduction of [1] inplace
   *        it is a specialization for binomial opinions.
   *        for the deduction the base rate of this, together with two conditional opinions must be available.
   * @param base_x
   * @param cond_1
   * @param cond_2
   * @return
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& deduction_(FloatT base_x, OpinionNoBase cond_1, OpinionNoBase cond_2)
    requires is_binomial<N>;

  /**
   * @brief applies the concept of deduction of [1] using a copy
   *        it is a specialization for binomial opinions.
   *        for the deduction the base rate of this, together with two conditional opinions must be available
   * @param base_x
   * @param cond_1
   * @param cond_2
   * @return
   */
  CUDA_AVAIL
  constexpr OpinionNoBase deduction(FloatT base_x, OpinionNoBase cond_1, OpinionNoBase cond_2) const
    requires is_binomial<N>;

  /**
   * @brief applies the concept of deduction of [1] inplace
   *        for the deduction the base rate of this, together with N conditional opinions must be available
   * @param base_x
   * @param conditionals
   * @return
   */
  CUDA_AVAIL
  constexpr OpinionNoBase& deduction_(BeliefType base_x, Array<N, OpinionNoBase> conditionals);

  /**
   * @brief applies the concept of deduction of [1] using a copy
   *        for the deduction the base rate of this, together with N conditional opinions must be available
   * @param base_x
   * @param conditionals
   * @return
   */
  CUDA_AVAIL
  constexpr OpinionNoBase deduction(BeliefType base_x, Array<N, OpinionNoBase> conditionals) const;

  /**
   * @brief multinomial Opinions can be reduced by accumulating belief masses of certain classes to one
   *        given the assignment of belief masses, this function accumulates the belief mass according to this
   * @tparam newN
   * @param instance_reduction
   * @return reduced opinion
   */
  template <std::size_t newN>
  CUDA_AVAIL constexpr OpinionNoBase<newN, FloatT> getReducedOpinion(Array<N, std::size_t> instance_reduction) const
    requires(not is_binomial<N> and newN < N);

  /**
   * @brief multinomial Opinions can be reduced by accumulating belief masses of certain classes to one
   *        given the assignment of belief masses, this function accumulates the belief mass according to this
   *        this impl. allows to use std::arrays directly
   * @tparam newN
   * @param instance_reduction
   * @return reduced opinion
   */
  template <std::size_t newN>
  CUDA_AVAIL constexpr OpinionNoBase<newN, FloatT>
  getReducedOpinion(std::array<std::size_t, N> instance_reduction) const
    requires(not is_binomial<N> and newN < N);

  /**
   * @brief equals operator, compares the sum of absolut differences between all belief masses to zero.
   * @param other
   * @return
   */
  CUDA_AVAIL
  constexpr bool operator==(const OpinionNoBase<N, FloatT>& other) const;

  /**
   * @brief converts opinion_no_base to a Dirichlet distribution preserving evidence and using a non informative prior
   */
  CUDA_AVAIL
  constexpr operator DirichletDistribution<N, FloatT>() const;

  /**
   * @brief generates a readable string containing the belief masses and the uncertainty
   * @return
   */
  explicit operator std::string() const;

  /**
   * @brief generates a readable string containing the belief masses and the uncertainty
   * @return
   */
  [[nodiscard]] std::string to_string() const;

protected:
  // belief distribution represents the evidence for each hypothesis of the considered domain
  BeliefType belief_masses_;
};

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>::OpinionNoBase(OpinionNoBase::BeliefType belief_masses)
  : belief_masses_{ belief_masses }
{
}

template <std::size_t N, typename FloatT>
template <typename... VALUES>
constexpr OpinionNoBase<N, FloatT>::OpinionNoBase(VALUES... values)
  requires(is_arithmetic_list<VALUES...> and sizeof...(VALUES) == N)
  : OpinionNoBase(BeliefType(static_cast<FloatT>(values)...))
{
}

template <std::size_t N, typename FloatT>
constexpr bool OpinionNoBase<N, FloatT>::is_valid() const
{
  bool valid_entries{ true };
  constexpr_for<0, N>(
      [&valid_entries, this](std::size_t idx) { valid_entries &= belief_masses_[idx] >= -EPS_v<FloatT>; });
  return valid_entries and belief_masses_.sum() < static_cast<FloatT>(1.0) + EPS_v<FloatT>;
}

template <std::size_t N, typename FloatT>
constexpr const FloatT& OpinionNoBase<N, FloatT>::belief() const
  requires is_binomial<N>
{
  return belief_masses_[0];
}

template <std::size_t N, typename FloatT>
constexpr FloatT& OpinionNoBase<N, FloatT>::belief()
  requires is_binomial<N>
{
  return belief_masses_[0];
}

template <std::size_t N, typename FloatT>
constexpr const FloatT& OpinionNoBase<N, FloatT>::disbelief() const
  requires is_binomial<N>
{
  return belief_masses_[1];
}

template <std::size_t N, typename FloatT>
constexpr FloatT& OpinionNoBase<N, FloatT>::disbelief()
  requires is_binomial<N>
{
  return belief_masses_[1];
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::complement() const
  requires is_binomial<N>
{
  BeliefType complement_belief_masses{ belief_masses_[1], belief_masses_[0] };
  return OpinionNoBase(complement_belief_masses);
}

template <std::size_t N, typename FloatT>
constexpr typename OpinionNoBase<N, FloatT>::BeliefType& OpinionNoBase<N, FloatT>::belief_masses()
{
  return belief_masses_;
}

template <std::size_t N, typename FloatT>
const constexpr typename OpinionNoBase<N, FloatT>::BeliefType& OpinionNoBase<N, FloatT>::belief_masses() const
{
  return belief_masses_;
}

template <std::size_t N, typename FloatT>
constexpr FloatT& OpinionNoBase<N, FloatT>::belief_mass(std::size_t idx)
{
  return belief_masses_[idx];
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::belief_mass(std::size_t idx) const
{
  return belief_masses_[idx];
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::uncertainty() const
{
  return static_cast<FloatT>(1.0) - belief_masses_.sum();
}

template <std::size_t N, typename FloatT>
constexpr Array<N, FloatT> OpinionNoBase<N, FloatT>::evidence() const
{
  Array<N, FloatT> belief_masses = belief_masses_;
  return belief_masses * (static_cast<FloatT>(SIZE) / uncertainty());
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::interpolate(OpinionNoBase other, FloatT interp_fac) const
{
  OpinionNoBase interp{ *this };

  FloatT this_fac = 1 - interp_fac;
  constexpr_for<0, N, 1>([&, this_fac](std::size_t idx) {
    interp.belief_masses_[idx] = this_fac * this->belief_masses_[idx] + interp_fac * other.belief_masses_[idx];
  });

  return interp;
}

template <std::size_t N, typename FloatT>
constexpr typename OpinionNoBase<N, FloatT>::BeliefType OpinionNoBase<N, FloatT>::NeutralBeliefDistr()
{
  return BeliefType{ static_cast<FloatT>(1.0) / N };
}

template <std::size_t N, typename FloatT>
constexpr typename OpinionNoBase<N, FloatT>::BeliefType OpinionNoBase<N, FloatT>::VacuousBeliefDistr()
{
  return OpinionNoBase().belief_masses_;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::NeutralBeliefOpinion()
{
  return OpinionNoBase{ NeutralBeliefDistr() };
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::VacuousBeliefOpinion()
{
  return {};
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::dissonance() const
{
  auto balance = [](FloatT a, FloatT b) {
    FloatT denom = a + b;
    if (denom < EPS_v<FloatT>)
    {
      return static_cast<FloatT>(0.);
    }
    FloatT diff = a - b;
    if (diff < 0)
    {
      return static_cast<FloatT>(1.) + diff;
    }
    return static_cast<FloatT>(1.) - diff;
  };

  FloatT dissonance = 0;
  FloatT belief_sum = belief_masses_.sum();

  constexpr_for<0, N>([this, balance, belief_sum, &dissonance](std::size_t i) {
    FloatT other_balanced{ 0. };
    constexpr_for<0, N>([this, i, balance, &other_balanced](std::size_t j) {
      if (i == j)
      {
        return;
      }
      other_balanced += belief_masses_[j] * balance(belief_masses_[i], belief_masses_[j]);
    });
    FloatT denom = belief_sum - belief_masses_[i];
    if (denom < EPS_v<FloatT>)
    {
      return;
    }
    dissonance += belief_masses_[i] * other_balanced / denom;
  });
  return dissonance;
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::getProbability() const
  requires is_binomial<N>
{
  FloatT denom = 1. - this->uncertainty();
  return belief_masses_[0] / denom;
}

template <std::size_t N, typename FloatT>
constexpr typename OpinionNoBase<N, FloatT>::BeliefType OpinionNoBase<N, FloatT>::getProbabilities() const
{
  FloatT denom = 1. - this->uncertainty();
  BeliefType probs;
  constexpr_for<0, N, 1, BeliefType&>(
      [this, denom] CUDA_AVAIL(std::size_t idx, BeliefType & probs_) { probs_[idx] = belief_masses_[idx] / denom; },
      probs);
  return probs;
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::getBinomialProjection(FloatT base_rate) const
  requires is_binomial<N>
{
  return belief() + uncertainty() * base_rate;
}

template <std::size_t N, typename FloatT>
constexpr typename OpinionNoBase<N, FloatT>::BeliefType
OpinionNoBase<N, FloatT>::getProjection(BeliefType base_rate) const
{
  BeliefType projection;
  FloatT uncert = uncertainty();
  constexpr_for<0, N, 1>([&](std::size_t idx) { projection[idx] = belief_masses_[idx] + uncert * base_rate[idx]; });
  return projection;
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::uncertainty_differential(OpinionNoBase other) const
{
  FloatT uncert{ this->uncertainty() };
  return uncert / (uncert + other.uncertainty());
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::degree_of_conflict(OpinionNoBase other,
                                                              FloatT base_rate,
                                                              FloatT base_rate_other) const
  requires is_binomial<N>
{
  FloatT proj_prob_distance =
      std::abs(this->getBinomialProjection(base_rate) - other.getBinomialProjection(base_rate_other));
  FloatT conjunctive_certainty = (1 - this->uncertainty()) * (1 - other.uncertainty());

  return proj_prob_distance * conjunctive_certainty;
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::degree_of_conflict(OpinionNoBase other,
                                                              BeliefType base_rate,
                                                              BeliefType base_rate_other) const
{
  if constexpr (is_binomial<N>)
  {
    return degree_of_conflict(other, base_rate.front(), base_rate_other.front());
  }
  FloatT proj_prob_distance{ 0. };
  BeliefType prob_this = getProjection(base_rate);
  BeliefType prob_other = other.getProjection(base_rate);

  constexpr_for<0, N, 1>([&](std::size_t idx) { proj_prob_distance += std::abs(prob_this[idx] - prob_other[idx]); });
  proj_prob_distance /= 2.;

  FloatT conjunctive_certainty = (1 - this->uncertainty()) * (1 - other.uncertainty());

  return proj_prob_distance * conjunctive_certainty;
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::degree_of_harmony(OpinionNoBase other,
                                                             FloatT base_rate,
                                                             FloatT base_rate_other) const
  requires is_binomial<N>
{
  FloatT proj_prob_distance =
      std::abs(this->getBinomialProjection(base_rate) - other.getBinomialProjection(base_rate_other));
  FloatT conjunctive_certainty = (1 - this->uncertainty()) * (1 - other.uncertainty());

  return (1 - proj_prob_distance) * conjunctive_certainty;
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::degree_of_harmony(OpinionNoBase other,
                                                             BeliefType base_rate,
                                                             BeliefType base_rate_other) const
{
  if constexpr (is_binomial<N>)
  {
    return degree_of_conflict(other, base_rate.front(), base_rate_other.front());
  }
  FloatT proj_prob_distance{ 0. };
  BeliefType prob_this = getProjection(base_rate);
  BeliefType prob_other = other.getProjection(base_rate);

  constexpr_for<0, N, 1>([&](std::size_t idx) { proj_prob_distance += std::abs(prob_this[idx] - prob_other[idx]); });
  proj_prob_distance /= 2.;

  FloatT conjunctive_certainty = (1 - this->uncertainty()) * (1 - other.uncertainty());

  return (1 - proj_prob_distance) * conjunctive_certainty;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::revise_trust_(FloatT degree_of_conflict,
                                                                            OpinionNoBase other)
  requires is_binomial<N>
{
  FloatT revision_factor = this->uncertainty_differential(other) * degree_of_conflict;
  return revise_trust_(revision_factor);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::revise_trust(FloatT degree_of_conflict,
                                                                          OpinionNoBase other) const
  requires is_binomial<N>
{
  return OpinionNoBase(*this).revise_trust_(degree_of_conflict, other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::revise_trust_(FloatT revision_factor)
  requires is_binomial<N>
{
  revision_factor = std::clamp(revision_factor, static_cast<FloatT>(-1.0), static_cast<FloatT>(1.0));

  if (revision_factor < 0)
  {
    revision_factor *= -1;
    belief_masses_[0] += (1 - belief_masses_[0]) * revision_factor;
    belief_masses_[1] *= (1 - revision_factor);
    return *this;
  }
  else
  {
    belief_masses_[0] *= (1 - revision_factor);
    belief_masses_[1] += (1 - belief_masses_[1]) * revision_factor;
    return *this;
  }
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::revise_trust(FloatT revision_factor) const
  requires is_binomial<N>
{
  return OpinionNoBase(*this).revise_trust_(revision_factor);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::multiply_(OpinionNoBase other,
                                                                        FloatT base_this,
                                                                        FloatT base_other)
  requires is_binomial<N>
{
  FloatT fac = (1 - base_this) * base_other * this->belief() * other.uncertainty() +
               base_this * (1 - base_other) * this->uncertainty() * other.belief();
  fac /= (1 - base_this * base_other);
  this->belief() = this->belief() * other.belief() + fac;
  this->disbelief() += other.disbelief() - this->disbelief() * other.disbelief();

  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::multiply(OpinionNoBase other,
                                                                      FloatT base_this,
                                                                      FloatT base_other) const
  requires is_binomial<N>
{
  return OpinionNoBase(*this).multiply_(other, base_this, base_other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::comultiply_(OpinionNoBase other,
                                                                          FloatT base_this,
                                                                          FloatT base_other)
  requires is_binomial<N>
{
  FloatT fac = base_this * (1 - base_other) * this->disbelief() * other.uncertainty() +
               (1 - base_this) * base_other * this->uncertainty() * other.disbelief();
  fac /= base_this + base_other - base_this * base_other;
  this->disbelief() = this->disbelief() * other.disbelief() + fac;
  this->belief() += other.belief() - this->belief() * other.belief();

  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::comultiply(OpinionNoBase other,
                                                                        FloatT base_this,
                                                                        FloatT base_other) const
  requires is_binomial<N>
{
  return OpinionNoBase(*this).comultiply_(other, base_this, base_other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::cum_fuse_(OpinionNoBase other)
{
  FloatT uncert_this = this->uncertainty();
  FloatT uncert_other = other.uncertainty();
  FloatT denom = uncert_this + uncert_other - uncert_this * uncert_other;

  if (std::abs(denom) < EPS_v<FloatT>)
  {
    // Jøsang suggests a boundary value consideration,
    // however, since no further information about the uncertainty values is available,
    // a simple mean is used instead
    constexpr_for<0, N, 1, OpinionNoBase&>(
        [this] CUDA_AVAIL(std::size_t idx, OpinionNoBase & other_) {
          belief_masses_[idx] = (belief_masses_[idx] + other_.belief_masses_[idx]) / 2.0;
        },
        other);
    return *this;
  }

  constexpr_for<0, N, 1>([this, other, denom, uncert_this, uncert_other] CUDA_AVAIL(std::size_t idx) {
    belief_masses_[idx] = (belief_masses_[idx] * uncert_other + other.belief_masses_[idx] * uncert_this) / denom;
  });
  return *this;
}
template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::cum_unfuse_(OpinionNoBase other)
{
  FloatT uncert_this = this->uncertainty();
  FloatT uncert_other = other.uncertainty();
  FloatT denom = uncert_other - uncert_this + uncert_other * uncert_this;

  if (std::abs(denom) < EPS_v<FloatT>)
  {
    // Jøsang suggests a boundary value consideration,
    // however, since no further information about the uncertainty values is available,
    // a simple mean is used instead
    constexpr_for<0, N, 1, OpinionNoBase&>(
        [this] CUDA_AVAIL(std::size_t idx, OpinionNoBase & other_) {
          belief_masses_[idx] = (belief_masses_[idx] + other_.belief_masses_[idx]) / 2.0;
        },
        other);
    return *this;
  }

  constexpr_for<0, N, 1>([this, other, denom, uncert_this, uncert_other] CUDA_AVAIL(std::size_t idx) {
    belief_masses_[idx] = (belief_masses_[idx] * uncert_other - other.belief_masses_[idx] * uncert_this) / denom;
  });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::cum_fuse(OpinionNoBase other) const
{
  return OpinionNoBase(*this).cum_fuse_(other);
}
template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::cum_unfuse(OpinionNoBase other) const
{
  return OpinionNoBase(*this).cum_unfuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr typename OpinionNoBase<N, FloatT>::BeliefType OpinionNoBase<N, FloatT>::harmony(OpinionNoBase other) const
{
  FloatT uncert_this = this->uncertainty();
  FloatT uncert_other = other.uncertainty();

  BeliefType harmony;

  constexpr_for<0, N, 1, BeliefType&>(
      [this, other, uncert_this, uncert_other] CUDA_AVAIL(std::size_t idx, BeliefType & harmony_) {
        harmony_[idx] = belief_masses_[idx] * uncert_other + other.belief_masses_[idx] * uncert_this +
                        belief_masses_[idx] * other.belief_masses_[idx];
      },
      harmony);
  return harmony;
}

template <std::size_t N, typename FloatT>
constexpr FloatT OpinionNoBase<N, FloatT>::conflict(OpinionNoBase other) const
{
  FloatT conflict{ 0. };
  for (std::size_t idx1{ 0 }; idx1 < N; ++idx1)
  {
    for (std::size_t idx2{ 0 }; idx2 < N; ++idx2)
    {
      if (idx1 == idx2)
      {
        continue;
      }
      conflict += this->belief_masses_[idx1] * other.belief_masses_[idx2];
    }
  }
  return conflict;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::bc_fuse_(OpinionNoBase other)
{
  BeliefType harmony = this->harmony(other);
  FloatT conflict = this->conflict(other);

  if (std::abs(1 - conflict) < EPS_v<FloatT>)
  {
    belief_masses_ = NeutralBeliefDistr();
    return *this;
  }

  FloatT normalizer = 1 - conflict;
  constexpr_for<0, N, 1>(
      [this, harmony, normalizer] CUDA_AVAIL(std::size_t idx) { belief_masses_[idx] = harmony[idx] / normalizer; });

  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::bc_fuse(OpinionNoBase other) const
{
  return OpinionNoBase(*this).bc_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::average_fuse_(OpinionNoBase other)
{
  FloatT uncert_this = this->uncertainty();
  FloatT uncert_other = other.uncertainty();
  FloatT denom = uncert_this + uncert_other;

  if (std::abs(denom) < EPS_v<FloatT>)
  {
    constexpr_for<0, N, 1>([&](std::size_t idx) {
      // Jøsang suggests a boundary value consideration,
      // however, since no further information about the uncertainty values is available,
      // a simple mean is used instead
      belief_masses_[idx] = (belief_masses_[idx] + other.belief_masses_[idx]) / 2.;
    });
    return *this;
  }

  constexpr_for<0, N, 1>([&, denom](std::size_t idx) {
    belief_masses_[idx] = (belief_masses_[idx] * uncert_other + other.belief_masses_[idx] * uncert_this) / denom;
  });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::average_fuse(OpinionNoBase other) const
{
  return OpinionNoBase(*this).average_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::average_unfuse_(OpinionNoBase other)
{
  FloatT uncert_this = this->uncertainty();
  FloatT uncert_other = other.uncertainty();
  FloatT denom = static_cast<FloatT>(2.0) * uncert_other - uncert_this;

  if (std::abs(denom) < EPS_v<FloatT>)
  {
    constexpr_for<0, N, 1>([&](std::size_t idx) {
      // Jøsang suggests a boundary value consideration,
      // however, since no further information about the uncertainty values is available,
      // a simple mean is used instead
      belief_masses_[idx] = (belief_masses_[idx] + other.belief_masses_[idx]) / 2.;
    });
    return *this;
  }

  constexpr_for<0, N, 1>([&, denom](std::size_t idx) {
    belief_masses_[idx] =
        (static_cast<FloatT>(2.0) * belief_masses_[idx] * uncert_other + other.belief_masses_[idx] * uncert_this) /
        denom;
  });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::average_unfuse(OpinionNoBase other) const
{
  return OpinionNoBase(*this).average_unfuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::cc_fuse_(OpinionNoBase other)
{
  FloatT uncert_this = this->uncertainty();
  FloatT uncert_other = other.uncertainty();

  BeliefType consensus{ 0. };
  FloatT consensus_sum{ 0. };
  BeliefType resA = belief_masses_;
  BeliefType resB = other.belief_masses_;

  BeliefType compromise{ 0 };
  FloatT compromise_sum{ 0. };

  constexpr_for<0, N, 1>([&](std::size_t idx) {
    FloatT consens = fminf(belief_masses_[idx], other.belief_masses_[idx]);
    consensus[idx] = consens;
    consensus_sum += consens;
    resA[idx] -= consensus[idx];
    resB[idx] -= consensus[idx];
  });

  constexpr_for<0, N, 1>([&](std::size_t idx) {
    // the part of the compromise with different hypotheses
    FloatT comp_different{ 0. };

    constexpr_for<0, N, 1>([&](std::size_t idx2) {
      if (idx == idx2)
      {
        return;
      }
      comp_different += resA[idx] * resB[idx2];
      comp_different += resA[idx2] * resB[idx];
    });

    compromise[idx] = resA[idx] * uncert_other +
                      resB[idx] * uncert_this
                      // when not dealing with hyperopinions,
                      // the second and third line sums can be merged leading to a single multiplication
                      + resA[idx] * resB[idx] + comp_different;

    compromise_sum += compromise[idx];
  });

  FloatT uncert_pre = uncert_this * uncert_other;

  if (fabs(compromise_sum) < EPS_v<FloatT>)
  {
    // compromise gets smaller with a decreasing amount of belief masses
    // thus return a vacuous opinion as a result
    belief_masses_ = VacuousBeliefDistr();
    return *this;
  }

  FloatT normalization = (1 - consensus_sum - uncert_pre) / compromise_sum;

  // merge consensus and compromise
  constexpr_for<0, N, 1>([this, consensus, compromise, normalization] CUDA_AVAIL(std::size_t idx) {
    belief_masses_[idx] = consensus[idx] + normalization * compromise[idx];
  });

  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::cc_fuse(OpinionNoBase other) const
{
  return OpinionNoBase(*this).cc_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::moment_matching_update_(BeliefType probabilities)
{
  auto distr = static_cast<DirichletDistribution<N, FloatT>>(*this);
  distr.moment_matching_update_(probabilities);
  *this = static_cast<OpinionNoBase>(distr);
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::moment_matching_update(BeliefType probabilities) const
{
  return OpinionNoBase(*this).moment_matching_update_(probabilities);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::wb_fuse_(OpinionNoBase other)
{
  FloatT uncert_this = this->uncertainty();
  FloatT uncert_other = other.uncertainty();
  FloatT denom = uncert_this + uncert_other - 2 * uncert_this * uncert_other;

  if (std::abs(denom) < EPS_v<FloatT>)
  {
    if (std::abs(uncert_this * uncert_other) < EPS_v<FloatT>)
    {
      // Jøsang suggests a boundary value consideration,
      // however, since no further information about the uncertainty values is available,
      // a simple mean is used instead
      constexpr_for<0, N, 1>([this, &other](std::size_t idx) {
        belief_masses_[idx] = (belief_masses_[idx] + other.belief_masses_[idx]) / 2.0;
      });
    }
    else
    {
      belief_masses_ = VacuousBeliefDistr();
    }
    return *this;
  }

  constexpr_for<0, N, 1>([this, &other, denom, uncert_this, uncert_other](std::size_t idx) {
    FloatT x_this = belief_masses_[idx];
    FloatT x_other = other.belief_masses_[idx];
    belief_masses_[idx] =
        (x_this * (1. - uncert_this) * uncert_other + x_other * (1. - uncert_other) * uncert_this) / denom;
  });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::wb_fuse(OpinionNoBase other) const
{
  return OpinionNoBase(*this).wb_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::trust_discount_(OpinionNoBase<2, FloatT> other,
                                                                              FloatT base_rate)
{
  return this->trust_discount_(other.getBinomialProjection(base_rate));
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::trust_discount(OpinionNoBase<2, FloatT> other,
                                                                            FloatT base_rate) const
{
  return OpinionNoBase(*this).trust_discount_(other, base_rate);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::trust_discount_(FloatT prop)
{
  constexpr_for<0, N, 1>([&, prop](std::size_t idx) constexpr -> void { belief_masses_[idx] *= prop; });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::trust_discount(FloatT prop) const
{
  return OpinionNoBase(*this).trust_discount_(prop);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::limited_trust_discount_(FloatT limit,
                                                                                      OpinionNoBase<2, FloatT> other,
                                                                                      FloatT base_rate)
{
  return this->limited_trust_discount_(limit, other.getBinomialProjection(base_rate));
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::limited_trust_discount(FloatT limit,
                                                                                    OpinionNoBase<2, FloatT> other,
                                                                                    FloatT base_rate) const
{
  return OpinionNoBase(*this).limited_trust_discount_(limit, other, base_rate);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::limited_trust_discount_(FloatT limit, FloatT prop)
{
  FloatT uncert = this->uncertainty();
  // updated uncertainy_mass may be limit at most
  FloatT min_prob{ (1 - limit) / (1 - uncert) };

  // as long as limit is ge zero, min_prob is ge zero,
  // but min_prob might be ge 1, in case that the uncert is too high beforehand
  // thus, std::min must be the outer check
  prop = std::min(static_cast<FloatT>(1.0), std::max(min_prob, prop));

  return trust_discount_(prop);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::limited_trust_discount(FloatT limit, FloatT prop) const
{
  return OpinionNoBase(*this).limited_trust_discount_(limit, prop);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::deduction_(FloatT base_x,
                                                                         OpinionNoBase cond_1,
                                                                         OpinionNoBase cond_2)
  requires is_binomial<N>
{
  FloatT a_y_nom = base_x * cond_1.belief() + (1 - base_x) * cond_2.belief();
  FloatT a_y_denom = 1 - (base_x * cond_1.uncertainty() + (1 - base_x) * cond_2.uncertainty());

  // if denom is too small (combination of vacuous conditional and base rate of x) simply take x's base rate
  // nothing is specifically defined in [1]
  FloatT a_y = base_x;
  if (std::abs(a_y_denom) > EPS_v<FloatT>)
  {
    a_y = a_y_nom / a_y_denom;
  }

  // projected probability distribution of the sub-simplex apex opinion
  FloatT P_apex_bel = base_x * cond_1.getBinomialProjection(a_y) + (1 - base_x) * cond_2.getBinomialProjection(a_y);

  // division by zero is no issue here
  // if a_y is zero, the respective result (inv) is filtered out by the min operation and vise versa
  FloatT uncert_apex = std::min((P_apex_bel - std::min(cond_1.belief(), cond_2.belief())) / a_y,
                                ((1 - P_apex_bel) - std::min(cond_1.disbelief(), cond_2.disbelief())) / (1 - a_y));

  FloatT uncert_y_x = uncert_apex - ((uncert_apex - cond_1.uncertainty()) * belief() +
                                     (uncert_apex - cond_2.uncertainty()) * disbelief());

  FloatT P_y_x = cond_1.getBinomialProjection(a_y) * getBinomialProjection(base_x) +
                 cond_2.getBinomialProjection(a_y) * (1 - getBinomialProjection(base_x));

  FloatT bel = P_y_x - a_y * uncert_y_x;

  belief() = bel;
  disbelief() = 1 - bel - uncert_y_x;
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::deduction(FloatT base_x,
                                                                       OpinionNoBase cond_1,
                                                                       OpinionNoBase cond_2) const
  requires is_binomial<N>
{
  return OpinionNoBase(*this).deduction_(base_x, cond_1, cond_2);
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& OpinionNoBase<N, FloatT>::deduction_(BeliefType base_x,
                                                                         Array<N, OpinionNoBase> conditionals)
{
  BeliefType a_y_nom;
  BeliefType a_y_denom;
  bool denom_near_zero{ false };
  constexpr_for<0, N, 1>([&](std::size_t y_idx) constexpr -> void {
    a_y_nom[y_idx] = 0.;
    a_y_denom[y_idx] = 0.;
    constexpr_for<0, N, 1>([&](std::size_t x_idx) constexpr -> void {
      a_y_nom[y_idx] += base_x[x_idx] * conditionals[x_idx].belief_mass(y_idx);
      a_y_denom[y_idx] += base_x[x_idx] * conditionals[x_idx].uncertainty();
    });
    denom_near_zero |= (1 - a_y_denom[y_idx]) < EPS_v<FloatT>;
  });

  // if denom is too small (combination of vacuous conditional and base rate of x) simply take x's base rate
  // nothing is specifically defined in [1]
  // (9.68) in [1]
  BeliefType a_y;
  if (denom_near_zero)
  {
    a_y = base_x;
  }
  else
  {
    constexpr_for<0, N, 1>(
        [&, a_y_nom, a_y_denom](std::size_t idx) constexpr -> void { a_y[idx] = a_y_nom[idx] / (1 - a_y_denom[idx]); });
  }

  BeliefType x_projection = getProjection(base_x);
  Array<N, BeliefType> cond_projections;
  constexpr_for<0, N, 1>(
      [&](std::size_t idx) constexpr -> void { cond_projections[idx] = conditionals[idx].getProjection(a_y); });

  // (9.70) in [1]
  BeliefType P_apex;
  constexpr_for<0, N, 1>([&](std::size_t y_idx) constexpr -> void {
    constexpr_for<0, N, 1>(
        [&](std::size_t x_idx) constexpr -> void { P_apex[y_idx] += base_x[x_idx] * cond_projections[x_idx][y_idx]; });
  });

  // division by zero is no issue here
  // if a_y is zero, the respective result (inv) is filtered out by the min operation and vise versa
  // (9.72) in [1]
  BeliefType uncertainties;
  constexpr_for<0, N, 1>([&](std::size_t y_idx) constexpr -> void {
    FloatT min_value =
        min<0, N>([&](std::size_t x_idx) constexpr -> FloatT { return conditionals[x_idx].belief_mass(y_idx); });
    uncertainties[y_idx] = (P_apex[y_idx] - min_value) / a_y[y_idx];
  });

  // (9.74) in [1]
  FloatT u_apex = min<0, N>([&](std::size_t y_idx) -> FloatT { return uncertainties[y_idx]; });

  // (9.75) in [1]
  FloatT u_y_x = u_apex * uncertainty();
  constexpr_for<0, N, 1>(
      [&](std::size_t x_idx) constexpr -> void { u_y_x += conditionals[x_idx].uncertainty() * belief_masses_[x_idx]; });

  // (9.61) in [1]
  BeliefType P_y_x;
  constexpr_for<0, N, 1>([&](std::size_t y_idx) constexpr -> void {
    constexpr_for<0, N, 1>([&](std::size_t x_idx) constexpr -> void {
      P_y_x[y_idx] += x_projection[x_idx] * cond_projections[x_idx][y_idx];
    });
  });

  // (9.77) in [1]
  BeliefType b_y_x;
  constexpr_for<0, N, 1>(
      [&](std::size_t y_idx) constexpr -> void { b_y_x[y_idx] = P_y_x[y_idx] - a_y[y_idx] * u_y_x; });

  // (9.78) in [1]
  belief_masses_ = b_y_x;
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT> OpinionNoBase<N, FloatT>::deduction(BeliefType base_x,
                                                                       Array<N, OpinionNoBase> conditionals) const
{
  return OpinionNoBase(*this).deduction_(base_x, conditionals);
}

template <std::size_t N, typename FloatT>
template <std::size_t newN>
constexpr OpinionNoBase<newN, FloatT>
OpinionNoBase<N, FloatT>::getReducedOpinion(Array<N, std::size_t> instance_reduction) const
  requires(not is_binomial<N> and newN < N)
{
  typename OpinionNoBase<newN, FloatT>::BeliefType out{};
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    auto new_hypotheses = instance_reduction[idx];
    assert(new_hypotheses < newN);
    out[new_hypotheses] += belief_masses_[idx];
  }

  return OpinionNoBase<newN, FloatT>{ out };
}

template <std::size_t N, typename FloatT>
template <std::size_t newN>
OpinionNoBase<newN, FloatT> constexpr OpinionNoBase<N, FloatT>::getReducedOpinion(
    std::array<std::size_t, N> instance_reduction) const
  requires(not is_binomial<N> and newN < N)
{
  return getReducedOpinion<newN>(static_cast<Array<N, std::size_t>>(instance_reduction));
}

template <std::size_t N, typename FloatT>
constexpr bool OpinionNoBase<N, FloatT>::operator==(const OpinionNoBase<N, FloatT>& other) const
{
  FloatT diff{ 0. };

  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    diff += std::abs(belief_masses_[idx] - other.belief_masses_[idx]);
  }
  return diff < EPS_v<FloatT>;
}

template <std::size_t N, typename FloatT>
inline std::ostream& operator<<(std::ostream& out, OpinionNoBase<N, FloatT> const& opinion)
{
  out << static_cast<std::string>(opinion);
  return out;
}

template <std::size_t N, typename FloatT>
std::string OpinionNoBase<N, FloatT>::to_string() const
{
  if constexpr (is_binomial<N>)
  {
    return std::string{ "[bel: " } + std::to_string(belief()) + "; disbel: " + std::to_string(disbelief()) +
           "; uncertainty: " + std::to_string(uncertainty()) + "]";
  }

  std::string out{ "[bel masses: " };
  for (const auto& mass : belief_masses_)
  {
    out += std::to_string(mass) + ", ";
  }
  out += "uncertainty: " + std::to_string(uncertainty()) + "]";

  return out;
}

template <std::size_t N, typename FloatT>
OpinionNoBase<N, FloatT>::operator std::string() const
{
  return this->to_string();
}

}  // namespace subjective_logic
