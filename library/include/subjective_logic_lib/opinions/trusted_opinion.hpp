#pragma once

// the reader is invited to refer to the following book as reference for the implementations within this file:
// Subjective Logic - A Formalism for Reasoning Under Uncertainty,
// Audun Jøsang, 2016, https://doi.org/10.1007/978-3-319-42337-1

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "subjective_logic_lib/util.hpp"
#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/opinions/opinion_no_base.hpp"

namespace subjective_logic
{

template <typename OpinionTemplate>
class TrustedOpinion
{
public:
  using OpinionT = OpinionTemplate;
  static constexpr std::size_t SIZE = OpinionT::SIZE;
  // define Float in two ways, so that it is compatible with Opinions and nice to use (FloatT)
  using FLOAT_t = typename OpinionT::FLOAT_t;
  using FloatT = FLOAT_t;
  using TrustT = Trust<FloatT>;

  /**
   * @brief accessor to simplify the access to the Opinions of TrustedOpinions stored in a vector
   *        opinions are copied during extraction
   * @param trusted_opinions
   * @return
   */
  static std::vector<OpinionT> extractOpinions(const std::vector<TrustedOpinion<OpinionT>>& trusted_opinions);

  /**
   * @brief accessor to simplify the access to the Opinions of TrustedOpinions stored in a vector
   *        opinions are NOT copied, but referenced during extraction
   * @param trusted_opinions
   * @return
   */
  static std::vector<std::reference_wrapper<OpinionT>>
  extractOpinionsRef(std::vector<TrustedOpinion>& trusted_opinions);

  /**
   * @brief accessor to simplify the access to the Trusts of TrustedOpinions stored in a vector
   *        opinions are copied during extraction
   * @param trusted_opinions
   * @return
   */
  static std::vector<TrustT> extractTrusts(const std::vector<TrustedOpinion>& trusted_opinions);

  /**
   * @brief accessor to simplify the access to the Trusts of TrustedOpinions stored in a vector
   *        opinions are NOT copied, but referenced during extraction
   * @param trusted_opinions
   * @return
   */
  static std::vector<std::reference_wrapper<TrustT>> extractTrustsRef(std::vector<TrustedOpinion>& trusted_opinions);

  /**
   * @brief accessor to simplify the access to the Opinions of TrustedOpinions stored in a vector
   *        opinions are copied during extraction
   * @param trusted_opinions
   * @return
   */
  static std::vector<OpinionT> extractDiscountedOpinions(const std::vector<TrustedOpinion>& trusted_opinions);

  /**
   * @brief default ctor leads to a vacuous opinion with vacuous trust
   */
  TrustedOpinion() = default;

  /**
   * @brief constructs a TrustedOpinion with the given trust and opinion
   * @param trust
   * @param opinion
   */
  TrustedOpinion(TrustT trust, OpinionT opinion);

  /**
   * @brief checks for both opinion and trust to be valid
   */
  constexpr bool is_valid() const;

  /**
   * @brief accessor to the trust part
   * @return
   */
  TrustT& trust();
  /**
   * @brief const accessor to the trust part
   * @return
   */
  const TrustT& trust() const;

  /**
   * @brief accessor to the opinion part
   * @return
   */
  OpinionT& opinion();
  /**
   * @brief const accessor to the opinion part
   * @return
   */
  const OpinionT& opinion() const;

  /**
   * @brief calculates and returns the opinion which is already discounted by the trust part of this class
   * @return
   */
  OpinionT discounted_opinion() const;

  /**
   * @brief applies the concept of trust revision described in [1] inplace
   *        here only the trust of this variable is updated with the given revision_factor
   * @param revision_factor
   * @return
   */
  TrustedOpinion& revise_trust_(FloatT revision_factor);

  /**
   * @brief applies the concept of trust revision described in [1]
   *        here only the trust of this variable is updated with the given revision_factor
   * @param revision_factor
   * @return
   */
  TrustedOpinion revise_trust(FloatT revision_factor) const;

  /**
   * @brief applies the concept of trust revision described in [1] inplace
   *        this means that both variables (this and other) are updated inplace and their new value is returned
   * @param other
   * @return
   */
  std::pair<TrustedOpinion&, TrustedOpinion&> revise_trust_(TrustedOpinion& other);
  /**
   * @brief applies the concept of trust revision described in [1]
   *        both results for (this and other) are returned
   * @param other
   * @return
   */
  std::pair<TrustedOpinion, TrustedOpinion> revise_trust(TrustedOpinion other) const;

  /**
   * @brief a trusted opinion equals another, if trust and opinion are equal
   * @param other
   * @return
   */
  CUDA_AVAIL
  constexpr bool operator==(const TrustedOpinion& other) const;

  /**
   * @brief generates a readable string containing the belief masses and the uncertainty for opinion and trust
   * @return
   */
  CUDA_AVAIL
  explicit operator std::string() const;

  /**
   * @brief generates a readable string containing the belief masses and the uncertainty for opinion and trust
   * @return
   */
  [[nodiscard]] std::string to_string() const;

protected:
  // trust_ represents the Trust on the opinion of this class,
  // it can be interpreted as functional trust on an entity with the belief opinion (opinion_)
  TrustT trust_;
  // opinion_ is a belief opinion on a variable of the domain at hand.
  OpinionT opinion_;
};

template <typename OpinionT>
std::vector<OpinionT>
TrustedOpinion<OpinionT>::extractOpinions(const std::vector<TrustedOpinion<OpinionT>>& trusted_opinions)
{
  std::vector<OpinionT> opinions;
  opinions.reserve(trusted_opinions.size());
  std::transform(trusted_opinions.cbegin(),
                 trusted_opinions.cend(),
                 std::back_inserter(opinions),
                 [](TrustedOpinion<OpinionT> top) { return top.opinion(); });
  return opinions;
}

template <typename OpinionT>
std::vector<std::reference_wrapper<OpinionT>>
TrustedOpinion<OpinionT>::extractOpinionsRef(std::vector<TrustedOpinion>& trusted_opinions)
{
  std::vector<std::reference_wrapper<OpinionT>> opinions;
  opinions.reserve(trusted_opinions.size());
  std::transform(trusted_opinions.begin(),
                 trusted_opinions.end(),
                 std::back_inserter(opinions),
                 [](TrustedOpinion<OpinionT>& top) -> OpinionT& { return top.opinion(); });
  return opinions;
}

template <typename OpinionT>
std::vector<typename TrustedOpinion<OpinionT>::TrustT>
TrustedOpinion<OpinionT>::extractTrusts(const std::vector<TrustedOpinion>& trusted_opinions)
{
  std::vector<TrustT> trusts;
  trusts.reserve(trusted_opinions.size());
  std::transform(trusted_opinions.cbegin(),
                 trusted_opinions.cend(),
                 std::back_inserter(trusts),
                 [](TrustedOpinion top) { return top.trust(); });
  return trusts;
}

template <typename OpinionT>
std::vector<std::reference_wrapper<typename TrustedOpinion<OpinionT>::TrustT>>
TrustedOpinion<OpinionT>::extractTrustsRef(std::vector<TrustedOpinion>& trusted_opinions)
{
  std::vector<std::reference_wrapper<TrustT>> trusts;
  trusts.reserve(trusted_opinions.size());
  std::transform(trusted_opinions.begin(),
                 trusted_opinions.end(),
                 std::back_inserter(trusts),
                 [](TrustedOpinion& top) -> TrustT& { return top.trust(); });
  return trusts;
}

template <typename OpinionT>
std::vector<typename TrustedOpinion<OpinionT>::OpinionT>
TrustedOpinion<OpinionT>::extractDiscountedOpinions(const std::vector<TrustedOpinion>& trusted_opinions)
{
  std::vector<OpinionT> discounted_opinions;
  discounted_opinions.reserve(trusted_opinions.size());
  std::transform(trusted_opinions.cbegin(),
                 trusted_opinions.cend(),
                 std::back_inserter(discounted_opinions),
                 [](TrustedOpinion top) { return top.discounted_opinion(); });
  return discounted_opinions;
}

template <typename OpinionT>
TrustedOpinion<OpinionT>::TrustedOpinion(TrustT trust, OpinionT opinion) : trust_{ trust }, opinion_{ opinion }
{
}

template <typename OpinionT>
constexpr bool TrustedOpinion<OpinionT>::is_valid() const
{
  return trust_.is_valid() and opinion_.is_valid();
}

template <typename OpinionT>
typename TrustedOpinion<OpinionT>::TrustT& TrustedOpinion<OpinionT>::trust()
{
  return trust_;
}

template <typename OpinionT>
const typename TrustedOpinion<OpinionT>::TrustT& TrustedOpinion<OpinionT>::trust() const
{
  return trust_;
}

template <typename OpinionT>
OpinionT& TrustedOpinion<OpinionT>::opinion()
{
  return opinion_;
}

template <typename OpinionT>
const OpinionT& TrustedOpinion<OpinionT>::opinion() const
{
  return opinion_;
}

template <typename OpinionT>
OpinionT TrustedOpinion<OpinionT>::discounted_opinion() const
{
  // generate trust projection explicitly to allow the same syntax for both Opinion types
  return opinion_.trust_discount(trust_.getBinomialProjection());
}

template <typename OpinionT>
std::pair<TrustedOpinion<OpinionT>, TrustedOpinion<OpinionT>>
TrustedOpinion<OpinionT>::revise_trust(TrustedOpinion other) const
{
  return TrustedOpinion(*this).revise_trust_(other);
}

template <typename OpinionT>
TrustedOpinion<OpinionT>& TrustedOpinion<OpinionT>::revise_trust_(FloatT revision_factor)
{
  trust_.revise_trust_(revision_factor);
  return *this;
}

template <typename OpinionT>
TrustedOpinion<OpinionT> TrustedOpinion<OpinionT>::revise_trust(FloatT revision_factor) const
{
  return TrustedOpinion{ *this }.revise_trust_(revision_factor);
}

template <typename OpinionT>
std::pair<TrustedOpinion<OpinionT>&, TrustedOpinion<OpinionT>&>
TrustedOpinion<OpinionT>::revise_trust_(TrustedOpinion& other)
{
  FloatT conflict = opinion_.degree_of_conflict(other.opinion_);

  FloatT revision_factor_this = trust_.uncertainty_differential(other.trust_) * conflict;
  FloatT revision_factor_other = other.trust_.uncertainty_differential(trust_) * conflict;

  trust_.revise_trust_(revision_factor_this);
  other.trust_.revise_trust_(revision_factor_other);

  return { *this, other };
}

template <typename OpinionT>
inline std::ostream& operator<<(std::ostream& out, TrustedOpinion<OpinionT> const& opinion)
{
  out << static_cast<std::string>(opinion);
  return out;
}

template <typename OpinionT>
constexpr bool TrustedOpinion<OpinionT>::operator==(const TrustedOpinion<OpinionT>& other) const
{
  return other.trust_ == trust_ and other.opinion_ == opinion_;
}

template <typename OpinionT>
std::string TrustedOpinion<OpinionT>::to_string() const
{
  return std::string{ "trust: " } + trust_.to_string() + " | opinion: " + opinion_.to_string();
}

template <typename OpinionT>
TrustedOpinion<OpinionT>::operator std::string() const
{
  return this->to_string();
}

}  // namespace subjective_logic
