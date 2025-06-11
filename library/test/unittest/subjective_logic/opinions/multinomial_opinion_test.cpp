#include "gtest/gtest.h"

#include "subjective_logic_lib/opinions/opinion.hpp"

namespace subjective_logic
{

using TestTypes = ::testing::Types<Opinion<3, float>, Opinion<6, float>, Opinion<3, double>, Opinion<6, double> >;

template <typename OpinionT>
class MultinomialOpinionTest : public ::testing::Test
{
  using FloatT = typename OpinionT::FLOAT_t;
  static constexpr std::size_t N = OpinionT::SIZE;

protected:
  void SetUp() override
  {
    variable_ = OpinionT::NeutralBeliefOpinion().trust_discount(1 - kUncertaintyMass_);
    variable_.prior_belief_masses() = prior_;
  }

  static constexpr FloatT kUncertaintyMass_{ 0.3 };
  static constexpr FloatT kEqualBeliefMass_{ (1.0 - kUncertaintyMass_) / N };

  OpinionT variable_;
  OpinionT::BeliefType prior_{ OpinionT::NeutralBeliefDistr() };
};

TYPED_TEST_SUITE(MultinomialOpinionTest, TestTypes);

TYPED_TEST(MultinomialOpinionTest, Ctor)
{
  for (auto const& value : this->variable_.belief_masses())
  {
    EXPECT_FLOAT_EQ(value, this->kEqualBeliefMass_);
  }

  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(this->variable_.belief_masses()[idx], this->variable_.belief_mass(idx));
  }

  auto base = this->variable_.as_no_base();
  auto prior = this->variable_.prior_belief_masses();

  TypeParam test{ base, prior };
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(test.belief_mass(idx), this->variable_.belief_mass(idx));
  }
}

TYPED_TEST(MultinomialOpinionTest, Uncertainty)
{
  ASSERT_FLOAT_EQ(this->variable_.uncertainty(), this->kUncertaintyMass_);
}

TYPED_TEST(MultinomialOpinionTest, GetProjection)
{
  typename TypeParam::FLOAT_t expected_projection =
      this->variable_.belief_masses()[0] + this->kUncertaintyMass_ * this->prior_[0];

  auto projection = this->variable_.getProjection();

  for (auto const& value : projection)
  {
    EXPECT_FLOAT_EQ(value, expected_projection);
  }
}

TYPED_TEST(MultinomialOpinionTest, Interpolate)
{
  using Float = typename TypeParam::FLOAT_t;
  constexpr std::size_t n = TypeParam::SIZE;

  Float interp_fac = 0.3;
  this->variable_ = TypeParam{};

  for (std::size_t idx{ 0 }; idx < n; ++idx)
  {
    TypeParam other{};
    other.belief_mass(idx) = 1.0;

    TypeParam test = this->variable_.interpolate(other, interp_fac);

    for (std::size_t idx_check{ 0 }; idx_check < n; ++idx_check)
    {
      if (idx == idx_check)
      {
        EXPECT_FLOAT_EQ(test.belief_mass(idx_check), interp_fac);
        continue;
      }
      EXPECT_FLOAT_EQ(test.belief_mass(idx_check), 0.);
    }
  }

  // test additionally one set for 2 belief_masses > 0
  TypeParam other{};
  Float test_belief_value = 0.25;
  other.belief_mass(0) = test_belief_value;
  other.belief_mass(1) = test_belief_value;

  TypeParam test = this->variable_.interpolate(other, interp_fac);

  EXPECT_FLOAT_EQ(test.belief_mass(0), test_belief_value * interp_fac);
  EXPECT_FLOAT_EQ(test.belief_mass(1), test_belief_value * interp_fac);

  for (std::size_t idx_check{ 2 }; idx_check < n; ++idx_check)
  {
    EXPECT_FLOAT_EQ(test.belief_mass(idx_check), 0.);
  }
}

TYPED_TEST(MultinomialOpinionTest, DegreeOfConflict)
{
  using Float = typename TypeParam::FLOAT_t;

  TypeParam first{};
  TypeParam second = first;

  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), 0.F);

  first.belief_masses()[0] = 1;
  second.belief_masses()[1] = 1;
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), 1.F);

  first.belief_masses()[0] = 0.5;
  second.belief_masses()[1] = 0.5;
  Float expected_doc = (0.5 + 0.5) / 2. * ((1 - 0.5) * (1 - 0.5));
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), expected_doc);
}

TYPED_TEST(MultinomialOpinionTest, CumulativeFusion)
{
  TypeParam neutral_element{};

  this->variable_.cum_fuse_(neutral_element);
  for (auto const& value : this->variable_.belief_masses())
  {
    EXPECT_FLOAT_EQ(value, this->kEqualBeliefMass_);
  }
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->kUncertaintyMass_);

  constexpr typename TypeParam::FLOAT_t var2_uncert{ 0.4 };
  typename TypeParam::FLOAT_t remaining_belief_mass{ 1 - var2_uncert };
  typename TypeParam::BeliefType var2_distr{};
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    var2_distr[idx] = remaining_belief_mass / 2.0;
    remaining_belief_mass /= 2.0;
  }
  var2_distr[TypeParam::SIZE - 1] += remaining_belief_mass;
  typename TypeParam::FLOAT_t checksum = std::accumulate(var2_distr.begin(), var2_distr.end(), 0.);
  EXPECT_FLOAT_EQ(checksum, 1 - var2_uncert);

  TypeParam var2{ var2_distr, TypeParam::NeutralBeliefDistr() };
  auto fused = this->variable_.cum_fuse(var2);
  for (auto const& value : this->variable_.belief_masses())
  {
    EXPECT_FLOAT_EQ(value, this->kEqualBeliefMass_);
  }
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->kUncertaintyMass_);

  typename TypeParam::FLOAT_t denom{ this->variable_.uncertainty() + var2.uncertainty() -
                                     this->variable_.uncertainty() * var2.uncertainty() };
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    typename TypeParam::FLOAT_t expected_belief{ this->variable_.belief_masses()[idx] * var2.uncertainty() +
                                                 var2.belief_masses()[idx] * this->variable_.uncertainty() };
    expected_belief /= denom;
    EXPECT_NEAR(fused.belief_masses()[idx], expected_belief, 1e-6);
  }
  typename TypeParam::FLOAT_t expected_uncertainty{ this->variable_.uncertainty() * var2.uncertainty() / denom };
  EXPECT_NEAR(fused.uncertainty(), expected_uncertainty, 1e-6);
}

TYPED_TEST(MultinomialOpinionTest, BeliefFusion)
{
  TypeParam neutral_element{};

  this->variable_.bc_fuse_(neutral_element);
  for (auto const& value : this->variable_.belief_masses())
  {
    EXPECT_NEAR(value, this->kEqualBeliefMass_, 1e-6);
  }
  EXPECT_NEAR(this->variable_.uncertainty(), this->kUncertaintyMass_, 1e-6);

  constexpr typename TypeParam::FLOAT_t var2_uncert{ 0.4 };
  typename TypeParam::FLOAT_t remaining_belief_mass{ 1 - var2_uncert };
  typename TypeParam::BeliefType var2_distr{};
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    var2_distr[idx] = remaining_belief_mass / 2.0;
    remaining_belief_mass /= 2.0;
  }
  var2_distr[TypeParam::SIZE - 1] += remaining_belief_mass;
  typename TypeParam::FLOAT_t checksum = std::accumulate(var2_distr.begin(), var2_distr.end(), 0.);
  ASSERT_FLOAT_EQ(checksum, 1 - var2_uncert);

  TypeParam var2{ var2_distr, TypeParam::NeutralBeliefDistr() };
  auto fused = this->variable_.bc_fuse(var2);
  for (auto const& value : this->variable_.belief_masses())
  {
    EXPECT_FLOAT_EQ(value, this->kEqualBeliefMass_);
  }
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->kUncertaintyMass_);

  typename TypeParam::FLOAT_t conflict{ 0. };
  std::vector<typename TypeParam::FLOAT_t> harmony_sum(TypeParam::SIZE);
  for (std::size_t idx_a{ 0 }; idx_a < TypeParam::SIZE; ++idx_a)
  {
    for (std::size_t idx_b{ 0 }; idx_b < TypeParam::SIZE; ++idx_b)
    {
      if (idx_a != idx_b)
      {
        conflict += this->variable_.belief_masses()[idx_a] * var2.belief_masses()[idx_b];
      }
      else
      {
        harmony_sum[idx_a] = this->variable_.belief_masses()[idx_a] * var2.belief_masses()[idx_b];
      }
    }
  }

  typename TypeParam::FLOAT_t denom = (1 - conflict);

  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    typename TypeParam::FLOAT_t expected_belief{ this->variable_.belief_masses()[idx] * var2.uncertainty() +
                                                 var2.belief_masses()[idx] * this->variable_.uncertainty() +
                                                 harmony_sum[idx] };
    expected_belief /= denom;

    EXPECT_FLOAT_EQ(fused.belief_masses()[idx], expected_belief);
  }
  typename TypeParam::FLOAT_t expected_uncertainty{ this->variable_.uncertainty() * var2.uncertainty() / denom };
  EXPECT_NEAR(fused.uncertainty(), expected_uncertainty, 1e-6);
}

TYPED_TEST(MultinomialOpinionTest, TrustDiscount)
{
  constexpr typename TypeParam::FLOAT_t kDiscountProp{ 0.8 };
  constexpr typename TypeParam::FLOAT_t kDiscountUncert{ 0.5 };
  typename Opinion<2, typename TypeParam::FLOAT_t>::BeliefType prior{ kDiscountProp, 1.0 - kDiscountProp };
  Opinion<2, typename TypeParam::FLOAT_t> kDiscountOpinion{ kDiscountProp * kDiscountUncert,
                                                            (1 - kDiscountProp) * kDiscountUncert,
                                                            kDiscountProp };

  auto var = this->variable_.trust_discount(kDiscountProp);
  auto var2 = this->variable_.trust_discount(kDiscountOpinion);

  EXPECT_FLOAT_EQ(var.uncertainty(), var2.uncertainty());

  typename TypeParam::FLOAT_t certainty = 1 - this->variable_.uncertainty();
  typename TypeParam::FLOAT_t expected_belief_sum = kDiscountProp * certainty;
  auto belief_sum =
      std::accumulate(var.belief_masses().begin(), var.belief_masses().end(), typename TypeParam::FLOAT_t{});
  EXPECT_FLOAT_EQ(belief_sum, expected_belief_sum);
  typename TypeParam::FLOAT_t expected_uncert = 1 - (kDiscountProp * certainty);
  EXPECT_FLOAT_EQ(var.uncertainty(), expected_uncert);
}

TYPED_TEST(MultinomialOpinionTest, ReducedOpinions)
{
  constexpr std::size_t newN{ 2 };
  std::array<std::size_t, TypeParam::SIZE> projection;
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    projection[idx] = idx % newN;
  }
  subjective_logic::Array projection_sl{ projection };

  auto reduced_opinion = this->variable_.template getReducedOpinion<newN>(projection);
  auto reduced_opinion_sl = this->variable_.template getReducedOpinion<newN>(projection_sl);

  auto uncertainty = reduced_opinion.uncertainty();

  subjective_logic::Array<newN, typename TypeParam::FLOAT_t> expected_belief_masses{};

  int remainder = TypeParam::SIZE % newN;
  for (std::size_t idx{ 0 }; idx < newN; ++idx)
  {
    expected_belief_masses[idx] = static_cast<int>(TypeParam::SIZE / newN);
    if (idx < remainder)
    {
      ++expected_belief_masses[idx];
    }
    expected_belief_masses[idx] *= this->kEqualBeliefMass_;
  }

  for (std::size_t idx{ 0 }; idx < newN; ++idx)
  {
    EXPECT_FLOAT_EQ(reduced_opinion.belief_masses()[idx], reduced_opinion_sl.belief_masses()[idx]);
    EXPECT_FLOAT_EQ(reduced_opinion.belief_masses()[idx], expected_belief_masses[idx]);
    EXPECT_FLOAT_EQ(reduced_opinion.prior_belief_masses()[idx], reduced_opinion_sl.prior_belief_masses()[idx]);
    EXPECT_FLOAT_EQ(reduced_opinion.prior_belief_masses()[idx], expected_belief_masses[idx] / (1 - uncertainty));
  }
}

TYPED_TEST(MultinomialOpinionTest, StringConversion)
{
  std::string s_1{ static_cast<std::string>(this->variable_) };
  std::string s_2{ this->variable_.to_string() };
  std::stringstream test{};
  test << this->variable_;
  std::string s_3{ test.str() };

  EXPECT_EQ(s_1, s_2);
  EXPECT_EQ(s_1, s_3);
}

}  // namespace subjective_logic
