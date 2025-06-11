#pragma once

// the reader is invited to refer to the following book as reference for the implementations within this file:
// [1] Subjective Logic - A Formalism for Reasoning Under Uncertainty,
// Audun Jøsang, 2016, https://doi.org/10.1007/978-3-319-42337-1

#include <array>
#include <iostream>
#include <numeric>

#include "subjective_logic_lib/opinions/opinion_no_base.hpp"

namespace subjective_logic
{
template <std::size_t N, typename FloatT>
class Opinion;
template <typename FloatT>
using Trust = Opinion<2, FloatT>;

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
class Opinion
{
public:
  using NoBaseType = OpinionNoBase<N, FloatT>;
  using FLOAT_t = typename NoBaseType::FLOAT_t;
  static constexpr std::size_t SIZE = NoBaseType::SIZE;

  using BeliefType = typename NoBaseType::BeliefType;

  /**
   * @brief creates a vacuous opinion, meaning that all belief masses are 0 and the uncertainty is 1
   */
  CUDA_AVAIL
  constexpr Opinion();

  /**
   * @brief creates an opinion using the given belief_mass and prior distributions, validity is not checked
   * @param belief_masses
   * @param prior
   */
  CUDA_AVAIL
  explicit constexpr Opinion(BeliefType belief_masses, BeliefType prior = Opinion::NeutralBeliefDistr());

  /**
   * @brief creates an opinion using the given an instance of OpinionNoBase and a prior distribution, validity is not
   * checked
   * @param opinion_no_base
   * @param prior
   */
  CUDA_AVAIL
  constexpr Opinion(NoBaseType opinion_no_base, BeliefType prior);

  /**
   * @brief allows to use the ctor with the correct number of arguments instead of requiring an initializer list
   *        when init an opinion like this, the prior is set to a neutral belief distr (see parent class)
   *        since defining multiple parameter packs is possible, but the resulting constructors would become quite
   *        confusing
   */
  template <typename... VALUES>
  CUDA_AVAIL constexpr explicit Opinion(VALUES... values)
    requires(is_arithmetic_list<VALUES...> and sizeof...(VALUES) == N);

  /**
   * @brief this ctor allows a convenient initialization of a binomial Opinion
   *        at least for binomial opinions the direct initialization of priors remains readable
   * @param belief_mass
   * @param disbelief_mass
   * @param prior
   */
  CUDA_AVAIL
  constexpr Opinion(FloatT belief_mass, FloatT disbelief_mass, FloatT prior)
    requires is_binomial<N>;

  /**
   * @brief this ctor allows a convenient initialization of a binomial Opinion
   *        at least for binomial opinions the direct initialization of priors remains readable
   * @param opinion_no_base
   * @param prior
   */
  CUDA_AVAIL
  constexpr Opinion(NoBaseType opinion_no_base, FloatT prior)
    requires is_binomial<N>;

  /**
   * @brief default copy ctor
   * @param other
   */
  constexpr Opinion(const Opinion& other) = default;
  /**
   * @brief default move ctor
   * @param other
   */
  constexpr Opinion(Opinion&& other) = default;

  /**
   * @brief default copy assignment
   * @param other
   */
  constexpr Opinion& operator=(const Opinion& other) = default;
  /**
   * @brief default move assignment
   * @param other
   */
  constexpr Opinion& operator=(Opinion&& other) = default;

  /**
   * @brief default dtor (not virtual!!!!, see description above)
   */
  ~Opinion() = default;

  /**
   * @brief checks for the validity of OpinionNoBase and sum of prior values to be 1.
   */
  constexpr bool is_valid() const;

  /**
   * @brief interface to the base class which allows public access of protected inherited members
   *        this function must be explicitly called such that a auto conversion should be prevented
   * @return
   */
  CUDA_AVAIL
  constexpr NoBaseType& as_no_base();
  /**
   * @brief interface to the base class which allows public access of protected inherited members
   *        this function must be explicitly called such that a auto conversion should be prevented
   * @return
   */
  CUDA_AVAIL
  constexpr const NoBaseType& as_no_base() const;

  /**
   * @brief creates a dogmatic trust opinion
   *        if an opinion is considered to be trust, the dogmatic trust is an opinion with belief = 1
   * @return
   */
  CUDA_AVAIL
  static constexpr Opinion DogmaticTrust()
    requires is_binomial<N>;

  /**
   * @brief creates a vacuous trust opinion
   *        if an opinion is considered to be trust, the vacuous trust is an opinion with belief = 0 and disbelief = 0
   * @return
   */
  CUDA_AVAIL
  static constexpr Opinion VacuousTrust()
    requires is_binomial<N>;

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
   * @brief makes function of OpinionNoBase available
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT& belief_mass(std::size_t idx);

  /**
   * @brief makes function of OpinionNoBase available
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT belief_mass(std::size_t idx) const;

  /**
   * @brief allows a convenient access to the prior belief_distribution
   * defined
   * @return
   */
  CUDA_AVAIL
  constexpr BeliefType& prior_belief_masses();

  /**
   * @brief allows a convenient access to the prior belief_distribution in a constant manner
   * defined
   * @return
   */
  CUDA_AVAIL
  constexpr const BeliefType& prior_belief_masses() const;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr inline FloatT uncertainty() const;
  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr inline Array<N, FloatT> evidence() const;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr inline const FloatT& belief() const
    requires is_binomial<N>;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr inline FloatT& belief()
    requires is_binomial<N>;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr inline const FloatT& disbelief() const
    requires is_binomial<N>;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr inline FloatT& disbelief()
    requires is_binomial<N>;

  /**
   * @brief convenient access to the belief of the prior belief distribution
   *        only available for binomial opinions
   * @return the belief of the prior binomial belief distribution
   */
  CUDA_AVAIL
  constexpr inline FloatT& prior_belief()
    requires is_binomial<N>;

  /**
   * @brief convenient access to the belief of the prior belief distribution in a constant manner
   *        only available for binomial opinions
   * @return the belief of the prior binomial belief distribution
   */
  CUDA_AVAIL
  constexpr inline const FloatT& prior_belief() const
    requires is_binomial<N>;

  /**
   * @brief convenient access to the disbelief of the prior disbelief distribution
   *        only available for binomial opinions
   * @return the disbelief of the prior binomial belief distribution
   */
  CUDA_AVAIL
  constexpr inline FloatT& prior_disbelief()
    requires is_binomial<N>;

  /**
   * @brief convenient access to the disbelief of the prior disbelief distribution in a constant manner
   *        only available for binomial opinions
   * @return the disbelief of the prior binomial belief distribution
   */
  CUDA_AVAIL
  constexpr inline const FloatT& prior_disbelief() const
    requires is_binomial<N>;

  /**
   * @brief makes function available from OpinionNoBase
   * @return
   */
  CUDA_AVAIL
  constexpr Opinion complement() const
    requires is_binomial<N>;

  /**
   * @brief calculates the dissonance of an opinion as proposed in [2]
   * @return the dissonance
   */
  CUDA_AVAIL
  constexpr FloatT dissonance() const;

  /**
   * @brief makes function available from OpinionNoBase
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT getProbability() const
    requires is_binomial<N>;

  /**
   * @brief extends function from OpinionNoBase
   * @return interpolated opinion
   */
  CUDA_AVAIL
  constexpr Opinion interpolate(Opinion other, FloatT interp_fac) const;

  /**
   * @brief makes function available from OpinionNoBase
   * @return
   */
  CUDA_AVAIL
  static constexpr BeliefType NeutralBeliefDistr();

  /**
   * @brief makes function available from OpinionNoBase
   * @return
   */
  CUDA_AVAIL
  static constexpr BeliefType VacuousBeliefDistr();

  /**
   * @brief creates an Opinion with a neutral belief distribution
   * @return
   */
  CUDA_AVAIL
  static constexpr Opinion NeutralBeliefOpinion();

  /**
   * @brief creates an Opinion with a vacuous belief distribution
   * @return
   */
  CUDA_AVAIL
  static constexpr Opinion VacuousBeliefOpinion();

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr BeliefType getProbabilities() const;

  /**
   * @brief convenient access to the first entry of the prior binomial belief distribution
   *        only available for binomial opinions
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT getBinomialPrior() const
    requires is_binomial<N>;

  /**
   * @brief convenient access to the first entry of the projected probability of binomial opinions
   *        only available for binomial opinions
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT getBinomialProjection() const
    requires is_binomial<N>;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr BeliefType getProjection() const;

  /**
   * @brief makes function available from protected inheritance
   *        changing the type to Opinion
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT uncertainty_differential(Opinion other) const;

  /**
   * @brief makes function available from protected inheritance
   *        changing the type to Opinion
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT degree_of_conflict(Opinion other) const;

  /**
   * @brief makes function available from protected inheritance
   *        changing the type to Opinion
   * @return
   */
  CUDA_AVAIL
  constexpr FloatT degree_of_harmony(Opinion other) const;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr Opinion& revise_trust_(FloatT degree_of_conflict, Opinion other)
    requires is_binomial<N>;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr Opinion revise_trust(FloatT degree_of_conflict, Opinion other) const
    requires is_binomial<N>;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr Opinion& revise_trust_(FloatT revision_factor)
    requires is_binomial<N>;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr Opinion revise_trust(FloatT revision_factor) const
    requires is_binomial<N>;

  /**
   * @brief extends the function of the base class inplace
   * @return multiplied opinion
   */
  CUDA_AVAIL
  constexpr Opinion& multiply_(Opinion other)
    requires is_binomial<N>;

  /**
   * @brief extends the function of the base class using a copy
   * @return multiplied opinion
   */
  CUDA_AVAIL
  constexpr Opinion multiply(Opinion other) const
    requires is_binomial<N>;

  /**
   * @brief extends the function of the base class inplace
   * @return comultiplied opinion
   */
  CUDA_AVAIL
  constexpr Opinion& comultiply_(Opinion other)
    requires is_binomial<N>;

  /**
   * @brief extends the function of the base class using a copy
   * @return comultiplied opinion
   */
  CUDA_AVAIL
  constexpr Opinion comultiply(Opinion other) const
    requires is_binomial<N>;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion& cum_fuse_(Opinion other);
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion cum_fuse(Opinion other) const;

  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr Opinion& cum_unfuse_(Opinion other);
  /**
   * @brief makes function available from protected inheritance
   * @return
   */
  CUDA_AVAIL
  constexpr Opinion cum_unfuse(Opinion other) const;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr BeliefType harmony(Opinion other) const;
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr FloatT conflict(Opinion other) const;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion& bc_fuse_(Opinion other);
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion bc_fuse(Opinion other) const;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion& average_fuse_(Opinion other);
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion average_fuse(Opinion other) const;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion& wb_fuse_(Opinion other);
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion wb_fuse(Opinion other) const;

  // I (wodtko) am not sure, if a use of this operator is senseful without considering hyperopinions.
  // the way it is implemented it simply considers all opinions to be singletons
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion& cc_fuse_(Opinion other);
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion cc_fuse(Opinion other) const;

  /**
   * @brief applies the concept of moment matching inline
   * @param probabilities
   * @return the updated opinion
   */
  CUDA_AVAIL
  constexpr Opinion& moment_matching_update_(BeliefType probabilities);
  /**
   * @brief applies the concept of moment matching using a copy
   * @param probabilities
   * @return the updated opinion
   */
  CUDA_AVAIL
  constexpr Opinion moment_matching_update(BeliefType probabilities) const;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion& trust_discount_(Trust<T> discount);
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion trust_discount(Trust<T> discount) const;
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion& trust_discount_(T prop)
    requires std::is_floating_point_v<T>;
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion trust_discount(T prop) const
    requires std::is_floating_point_v<T>;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion& limited_trust_discount_(FloatT limit, Trust<T> discount);
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion limited_trust_discount(FloatT limit, Trust<T> discount) const;
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion& limited_trust_discount_(FloatT limit, T prop)
    requires std::is_floating_point_v<T>;
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  template <typename T>
  CUDA_AVAIL constexpr Opinion limited_trust_discount(FloatT limit, T prop) const
    requires std::is_floating_point_v<T>;

  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion& deduction_(Opinion cond_bel, Opinion cond_dis)
    requires is_binomial<N>;
  /**
   * @brief extends the function of the base class, since the prior is implicitly available
   * @return the projectedProbability
   */
  CUDA_AVAIL
  constexpr Opinion deduction(Opinion cond_bel, Opinion cond_dis) const
    requires is_binomial<N>;

  /**
   * @brief multinomial Opinions can be reduced by accumulating belief masses of certain classes to one
   *        given the assignment of belief masses, this function accumulates the belief mass according to this
   *        prior belief masses are accumulated respectively
   * @tparam newN
   * @param instance_reduction
   * @return reduced opinion
   */
  template <std::size_t newN>
  CUDA_AVAIL constexpr Opinion<newN, FloatT> getReducedOpinion(Array<N, std::size_t> instance_reduction) const
    requires(not is_binomial<N> and newN < N);

  /**
   * @brief multinomial Opinions can be reduced by accumulating belief masses of certain classes to one
   *        given the assignment of belief masses, this function accumulates the belief mass according to this
   *        prior belief masses are accumulated respectively
   *        this impl. allows to use std::arrays directly
   * @tparam newN
   * @param instance_reduction
   * @return reduced opinion
   */
  template <std::size_t newN>
  CUDA_AVAIL constexpr Opinion<newN, FloatT> getReducedOpinion(std::array<std::size_t, N> instance_reduction) const
    requires(not is_binomial<N> and newN < N);
  /**
   * @brief checks base class and prior similarity. base class similarity is thereby checked first
   * @param other
   * @return true, if other is equal to this by less than eps
   */
  CUDA_AVAIL
  constexpr bool operator==(const Opinion& other) const;

  /**
   * @brief converts opinion to a Dirichlet distribution while preserving prior and evidence
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
  NoBaseType opinion_no_base_;
  // actually the prior would only require N-1 values, since it must sum up to 1
  // however, the implementation seems to be simpler when using a "full" prior
  BeliefType prior_;
};

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>::Opinion() : opinion_no_base_{}
{
  prior_ = OpinionNoBase<N, FloatT>::NeutralBeliefDistr();
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>::Opinion(BeliefType belief_masses, BeliefType prior)
  : opinion_no_base_{ belief_masses }, prior_{ prior }
{
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>::Opinion(NoBaseType opinion_no_base, BeliefType prior)
  : opinion_no_base_{ opinion_no_base }, prior_{ prior }
{
}

template <std::size_t N, typename FloatT>
template <typename... VALUES>
constexpr Opinion<N, FloatT>::Opinion(VALUES... values)
  requires(is_arithmetic_list<VALUES...> and sizeof...(VALUES) == N)
  : Opinion(BeliefType{ static_cast<FloatT>(values)... }, OpinionNoBase<N, FloatT>::NeutralBeliefDistr())
{
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>::Opinion(NoBaseType opinion_no_base, FloatT prior)
  requires is_binomial<N>
  : Opinion(opinion_no_base.belief(), opinion_no_base.disbelief(), prior)
{
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>::Opinion(FloatT belief_mass, FloatT disbelief_mass, FloatT prior)
  requires is_binomial<N>
  : Opinion{ BeliefType(belief_mass, disbelief_mass), BeliefType(prior, 1 - prior) }
{
}

template <std::size_t N, typename FloatT>
constexpr bool Opinion<N, FloatT>::is_valid() const
{
  bool base_valid = opinion_no_base_.is_valid();
  bool valid_entries{ true };
  constexpr_for<0, N>([&valid_entries, this](std::size_t idx) { valid_entries &= prior_[idx] >= -EPS_v<FloatT>; });
  auto prior_sum = prior_.sum();

  return base_valid and valid_entries and prior_sum > 1.0 - EPS_v<FloatT> and prior_sum < 1.0 + EPS_v<FloatT>;
}

template <std::size_t N, typename FloatT>
constexpr inline FloatT Opinion<N, FloatT>::uncertainty() const
{
  return opinion_no_base_.uncertainty();
}
template <std::size_t N, typename FloatT>
constexpr inline Array<N, FloatT> Opinion<N, FloatT>::evidence() const
{
  return opinion_no_base_.evidence();
}

template <std::size_t N, typename FloatT>
constexpr inline const FloatT& Opinion<N, FloatT>::belief() const
  requires is_binomial<N>
{
  return opinion_no_base_.belief();
}

template <std::size_t N, typename FloatT>
constexpr inline FloatT& Opinion<N, FloatT>::belief()
  requires is_binomial<N>
{
  return opinion_no_base_.belief();
}

template <std::size_t N, typename FloatT>
constexpr inline const FloatT& Opinion<N, FloatT>::disbelief() const
  requires is_binomial<N>
{
  return opinion_no_base_.disbelief();
}

template <std::size_t N, typename FloatT>
constexpr inline FloatT& Opinion<N, FloatT>::disbelief()
  requires is_binomial<N>
{
  return opinion_no_base_.disbelief();
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::complement() const
  requires is_binomial<N>
{
  return Opinion{ opinion_no_base_.complement().belief_masses(), prior_ };
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::dissonance() const
{
  return opinion_no_base_.dissonance();
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::interpolate(Opinion other, FloatT interp_fac) const
{
  OpinionNoBase interp_base = this->opinion_no_base_.interpolate(other.opinion_no_base_, interp_fac);
  decltype(prior_) interp_prior{};

  FloatT this_fac = 1 - interp_fac;
  constexpr_for<0, N, 1>([&, this_fac](std::size_t idx) {
    interp_prior[idx] = this_fac * this->prior_[idx] + interp_fac * other.prior_[idx];
  });

  return { interp_base, interp_prior };
}

template <std::size_t N, typename FloatT>
constexpr typename Opinion<N, FloatT>::BeliefType Opinion<N, FloatT>::NeutralBeliefDistr()
{
  return NoBaseType::NeutralBeliefDistr();
}

template <std::size_t N, typename FloatT>
constexpr typename Opinion<N, FloatT>::BeliefType Opinion<N, FloatT>::VacuousBeliefDistr()
{
  return NoBaseType::VacuousBeliefDistr();
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::NeutralBeliefOpinion()
{
  return Opinion{ NoBaseType::NeutralBeliefDistr(), NoBaseType::NeutralBeliefDistr() };
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::VacuousBeliefOpinion()
{
  return Opinion{ NoBaseType::VacuousBeliefDistr(), NoBaseType::VacuousBeliefDistr() };
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::getProbability() const
  requires is_binomial<N>
{
  return opinion_no_base_.getProbability();
}

template <std::size_t N, typename FloatT>
constexpr typename Opinion<N, FloatT>::BeliefType Opinion<N, FloatT>::getProbabilities() const
{
  return opinion_no_base_.getProbabilities();
}

template <std::size_t N, typename FloatT>
constexpr OpinionNoBase<N, FloatT>& Opinion<N, FloatT>::as_no_base()
{
  return opinion_no_base_;
}
template <std::size_t N, typename FloatT>
constexpr const OpinionNoBase<N, FloatT>& Opinion<N, FloatT>::as_no_base() const
{
  return opinion_no_base_;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::DogmaticTrust()
  requires is_binomial<N>
{
  return Opinion<N, FloatT>{ 1., 0. };
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::VacuousTrust()
  requires is_binomial<N>
{
  return Opinion<N, FloatT>{ 0., 0. };
}

template <std::size_t N, typename FloatT>
constexpr typename Opinion<N, FloatT>::BeliefType& Opinion<N, FloatT>::belief_masses()
{
  return opinion_no_base_.belief_masses();
}

template <std::size_t N, typename FloatT>
constexpr const typename Opinion<N, FloatT>::BeliefType& Opinion<N, FloatT>::belief_masses() const
{
  return opinion_no_base_.belief_masses();
}

template <std::size_t N, typename FloatT>
constexpr FloatT& Opinion<N, FloatT>::belief_mass(std::size_t idx)
{
  return opinion_no_base_.belief_mass(idx);
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::belief_mass(std::size_t idx) const
{
  return opinion_no_base_.belief_mass(idx);
}

template <std::size_t N, typename FloatT>
constexpr typename Opinion<N, FloatT>::BeliefType& Opinion<N, FloatT>::prior_belief_masses()
{
  return this->prior_;
}

template <std::size_t N, typename FloatT>
constexpr const typename Opinion<N, FloatT>::BeliefType& Opinion<N, FloatT>::prior_belief_masses() const
{
  return this->prior_;
}

template <std::size_t N, typename FloatT>
constexpr FloatT& Opinion<N, FloatT>::prior_belief()
  requires is_binomial<N>
{
  return prior_[0];
}

template <std::size_t N, typename FloatT>
constexpr const FloatT& Opinion<N, FloatT>::prior_belief() const
  requires is_binomial<N>
{
  return prior_[0];
}

template <std::size_t N, typename FloatT>
constexpr FloatT& Opinion<N, FloatT>::prior_disbelief()
  requires is_binomial<N>
{
  return prior_[1];
}

template <std::size_t N, typename FloatT>
constexpr const FloatT& Opinion<N, FloatT>::prior_disbelief() const
  requires is_binomial<N>
{
  return prior_[1];
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::getBinomialPrior() const
  requires is_binomial<N>
{
  return prior_[0];
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::getBinomialProjection() const
  requires is_binomial<N>
{
  return opinion_no_base_.getBinomialProjection(getBinomialPrior());
}

template <std::size_t N, typename FloatT>
constexpr typename Opinion<N, FloatT>::BeliefType Opinion<N, FloatT>::getProjection() const
{
  return opinion_no_base_.getProjection(prior_);
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::uncertainty_differential(Opinion other) const
{
  return opinion_no_base_.uncertainty_differential(other.opinion_no_base_);
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::degree_of_conflict(Opinion other) const
{
  if constexpr (is_binomial<N>)
  {
    return opinion_no_base_.degree_of_conflict(
        other.opinion_no_base_, this->getBinomialPrior(), other.getBinomialPrior());
  }
  return opinion_no_base_.degree_of_conflict(other.opinion_no_base_, this->prior_, other.prior_);
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::degree_of_harmony(Opinion<N, FloatT> other) const
{
  if constexpr (is_binomial<N>)
  {
    return opinion_no_base_.degree_of_harmony(
        other.opinion_no_base_, this->getBinomialPrior(), other.getBinomialPrior());
  }
  return opinion_no_base_.degree_of_harmony(other.opinion_no_base_, this->prior_, other.prior_);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::revise_trust_(FloatT degree_of_conflict, Opinion other)
  requires is_binomial<N>
{
  opinion_no_base_.revise_trust_(degree_of_conflict, other.opinion_no_base_);
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::revise_trust(FloatT degree_of_conflict, Opinion other) const
  requires is_binomial<N>
{
  return Opinion(*this).revise_trust_(degree_of_conflict, other);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::revise_trust_(FloatT revision_factor)
  requires is_binomial<N>
{
  opinion_no_base_.revise_trust_(revision_factor);
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::revise_trust(FloatT revision_factor) const
  requires is_binomial<N>
{
  return Opinion(*this).revise_trust_(revision_factor);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::multiply_(Opinion<N, FloatT> other)
  requires is_binomial<N>
{
  this->opinion_no_base_.multiply_(other.opinion_no_base_, this->prior_belief(), other.prior_belief());
  this->prior_belief() = this->prior_belief() * other.prior_belief();
  this->prior_disbelief() = 1.0 - this->prior_belief();

  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::multiply(Opinion<N, FloatT> other) const
  requires is_binomial<N>
{
  return Opinion(*this).multiply_(other);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::comultiply_(Opinion<N, FloatT> other)
  requires is_binomial<N>
{
  this->opinion_no_base_.comultiply_(other.opinion_no_base_, this->prior_belief(), other.prior_belief());
  this->prior_belief() += other.prior_belief() - this->prior_belief() * other.prior_belief();
  this->prior_disbelief() = 1.0 - this->prior_belief();

  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::comultiply(Opinion<N, FloatT> other) const
  requires is_binomial<N>
{
  return Opinion(*this).comultiply_(other);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::cum_fuse_(Opinion other)
{
  FloatT uncert_this = uncertainty();
  FloatT uncert_other = other.uncertainty();

  opinion_no_base_.cum_fuse_(other.opinion_no_base_);

  FloatT denom = uncert_this + uncert_other - 2. * uncert_this * uncert_other;
  if (std::abs(denom) < EPS_v<FloatT>)
  {
    constexpr_for<0, N, 1>([this, other](std::size_t idx) { prior_[idx] = (prior_[idx] + other.prior_[idx]) / 2.; });
    return *this;
  }
  constexpr_for<0, N, 1>([this, other, denom, uncert_this, uncert_other](std::size_t idx) {
    prior_[idx] = (prior_[idx] * uncert_other + other.prior_[idx] * uncert_this -
                   (prior_[idx] + other.prior_[idx]) * uncert_this * uncert_other) /
                  denom;
  });
  return *this;
}
template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::cum_unfuse_(Opinion other)
{
  opinion_no_base_.cum_unfuse_(other.opinion_no_base_);
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::cum_fuse(Opinion other) const
{
  return Opinion(*this).cum_fuse_(other);
}
template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::cum_unfuse(Opinion other) const
{
  return Opinion(*this).cum_unfuse_(other);
}
template <std::size_t N, typename FloatT>
constexpr typename Opinion<N, FloatT>::BeliefType Opinion<N, FloatT>::harmony(Opinion other) const
{
  return opinion_no_base_.harmony(other.opinion_no_base_);
}

template <std::size_t N, typename FloatT>
constexpr FloatT Opinion<N, FloatT>::conflict(Opinion other) const
{
  return opinion_no_base_.conflict(other.opinion_no_base_);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::bc_fuse_(Opinion other)
{
  FloatT uncert_this = uncertainty();
  FloatT uncert_other = other.uncertainty();

  opinion_no_base_.bc_fuse_(other.opinion_no_base_);

  FloatT denom = 2 - uncert_this - uncert_other;
  if (std::abs(denom) < EPS_v<FloatT>)
  {
    constexpr_for<0, N, 1>([&](std::size_t idx) { prior_[idx] = (prior_[idx] + other.prior_[idx]) / 2.; });
    return *this;
  }
  constexpr_for<0, N, 1>([&, uncert_this, uncert_other, denom](std::size_t idx) {
    prior_[idx] = (prior_[idx] * (1. - uncert_this) + other.prior_[idx] * (1. - uncert_other)) / denom;
  });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::bc_fuse(Opinion other) const
{
  return Opinion(*this).bc_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::average_fuse_(Opinion other)
{
  opinion_no_base_.average_fuse_(other.opinion_no_base_);
  constexpr_for<0, N, 1>(
      [&](std::size_t idx) constexpr -> void { prior_[idx] = (prior_[idx] + other.prior_[idx]) / 2.; });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::average_fuse(Opinion other) const
{
  return Opinion(*this).average_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::wb_fuse_(Opinion other)
{
  FloatT uncert_this = uncertainty();
  FloatT uncert_other = other.uncertainty();

  opinion_no_base_.wb_fuse_(other.opinion_no_base_);

  FloatT denom = 2 - uncert_this - uncert_other;
  if (std::abs(denom) < EPS_v<FloatT>)
  {
    constexpr_for<0, N, 1>([&](std::size_t idx) { prior_[idx] = (prior_[idx] + other.prior_[idx]) / 2.; });
    return *this;
  }

  constexpr_for<0, N, 1>([&, uncert_this, uncert_other, denom](std::size_t idx) {
    prior_[idx] = (prior_[idx] * (1 - uncert_this) + other.prior_[idx] * (1 - uncert_other)) / denom;
  });
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::wb_fuse(Opinion other) const
{
  return Opinion(*this).wb_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::cc_fuse_(Opinion other)
{
  // the respective prior is not updated according to [1]
  opinion_no_base_.cc_fuse_(other.opinion_no_base_);
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::cc_fuse(Opinion other) const
{
  return Opinion(*this).cc_fuse_(other);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::moment_matching_update_(BeliefType probabilities)
{
  auto distr = static_cast<DirichletDistribution<N, FloatT>>(*this);
  distr.moment_matching_update_(probabilities);
  *this = static_cast<Opinion>(distr);
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::moment_matching_update(BeliefType probabilities) const
{
  return Opinion(*this).moment_matching_update_(probabilities);
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::trust_discount_(Trust<T> discount)
{
  opinion_no_base_.trust_discount_(static_cast<FloatT>(discount.getBinomialProjection()));
  return *this;
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::trust_discount(Trust<T> discount) const
{
  return Opinion(*this).trust_discount_(discount);
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::trust_discount_(T prop)
  requires std::is_floating_point_v<T>
{
  opinion_no_base_.trust_discount_(prop);
  return *this;
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::trust_discount(T prop) const
  requires std::is_floating_point_v<T>
{
  return Opinion(*this).trust_discount_(prop);
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::limited_trust_discount_(FloatT limit, Trust<T> discount)
{
  opinion_no_base_.limited_trust_discount_(limit, static_cast<FloatT>(discount.getBinomialProjection()));
  return *this;
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::limited_trust_discount(FloatT limit, Trust<T> discount) const
{
  return Opinion(*this).limited_trust_discount_(limit, discount);
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::limited_trust_discount_(FloatT limit, T prop)
  requires std::is_floating_point_v<T>
{
  opinion_no_base_.limited_trust_discount_(limit, prop);
  return *this;
}

template <std::size_t N, typename FloatT>
template <typename T>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::limited_trust_discount(FloatT limit, T prop) const
  requires std::is_floating_point_v<T>
{
  return Opinion(*this).limited_trust_discount_(limit, prop);
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT>& Opinion<N, FloatT>::deduction_(Opinion cond_bel, Opinion cond_dis)
  requires is_binomial<N>
{
  FloatT a_y_nom = prior_belief() * cond_bel.belief() + prior_disbelief() * cond_dis.belief();
  FloatT a_y_denom = 1 - (prior_belief() * cond_bel.uncertainty() + prior_disbelief() * cond_dis.uncertainty());
  FloatT a_y = a_y_nom / a_y_denom;

  opinion_no_base_.deduction_(getBinomialPrior(), cond_bel.opinion_no_base_, cond_dis.opinion_no_base_);

  prior_ = BeliefType{ a_y, 1 - a_y };
  return *this;
}

template <std::size_t N, typename FloatT>
constexpr Opinion<N, FloatT> Opinion<N, FloatT>::deduction(Opinion cond_bel, Opinion cond_dis) const
  requires is_binomial<N>
{
  return Opinion(*this).deduction_(cond_bel, cond_dis);
}

template <std::size_t N, typename FloatT>
template <std::size_t newN>
constexpr Opinion<newN, FloatT> Opinion<N, FloatT>::getReducedOpinion(Array<N, std::size_t> instance_reduction) const
  requires(not is_binomial<N> and newN < N)
{
  typename OpinionNoBase<newN, FloatT>::BeliefType out_belief{};
  typename OpinionNoBase<newN, FloatT>::BeliefType out_prior{};

  constexpr_for<0, N, 1>([&](std::size_t idx) -> void {
    auto new_hypotheses = instance_reduction[idx];
    assert(new_hypotheses < newN);
    out_belief[new_hypotheses] += belief_masses()[idx];
    out_prior[new_hypotheses] += prior_belief_masses()[idx];
  });

  return Opinion<newN, FloatT>{ out_belief, out_prior };
}

template <std::size_t N, typename FloatT>
template <std::size_t newN>
Opinion<newN, FloatT> constexpr Opinion<N, FloatT>::getReducedOpinion(
    std::array<std::size_t, N> instance_reduction) const
  requires(not is_binomial<N> and newN < N)
{
  return getReducedOpinion<newN>(static_cast<Array<N, std::size_t>>(instance_reduction));
}

template <std::size_t N, typename FloatT>
constexpr bool Opinion<N, FloatT>::operator==(const Opinion& other) const
{
  FloatT diff{ 0. };
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    diff += std::abs(prior_[idx] - other.prior_[idx]);
  }
  return diff < EPS_v<FloatT> and opinion_no_base_ == other.opinion_no_base_;
}

template <std::size_t N, typename FloatT>
inline std::ostream& operator<<(std::ostream& out, Opinion<N, FloatT> const& opinion)
{
  out << static_cast<std::string>(opinion);
  return out;
}

template <std::size_t N, typename FloatT>
std::string Opinion<N, FloatT>::to_string() const
{
  if constexpr (is_binomial<N>)
  {
    std::string opinion = std::string{ "[bel: " } + std::to_string(belief()) +
                          "; disbel: " + std::to_string(disbelief()) +
                          "; uncertainty: " + std::to_string(uncertainty()) + "]";
    std::string prior = std::string{ "[bel: " } + std::to_string(prior_belief()) +
                        "; disbel: " + std::to_string(prior_disbelief()) + "]";
    return std::string{ "opinion: " } + opinion + " | prior: " + prior;
  }

  std::string opinion{ "[bel masses: " };
  for (const auto& mass : this->belief_masses())
  {
    opinion += std::to_string(mass) + ", ";
  }
  opinion += "uncertainty: " + std::to_string(uncertainty()) + "]";

  std::string prior{ "[bel masses: " };
  for (const auto& mass : this->prior_)
  {
    prior += std::to_string(mass) + ", ";
  }
  prior += "]";

  return std::string{ "opinion: " } + opinion + " | prior: " + prior;
}

template <std::size_t N, typename FloatT>
Opinion<N, FloatT>::operator std::string() const
{
  return this->to_string();
}

}  // namespace subjective_logic
