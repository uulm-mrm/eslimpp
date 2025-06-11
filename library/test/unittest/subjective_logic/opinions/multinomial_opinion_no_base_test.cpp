#include "gtest/gtest.h"

#include <numeric>
#include "subjective_logic_lib/opinions/opinion_no_base.hpp"

namespace subjective_logic
{

TEST(MultinomialOpinionNoBaseTest, JosangExampleCCFuse)
{
  OpinionNoBase<3, double> op_a(0.99, 0.01, 0.00);
  OpinionNoBase<3, double> op_b(0.00, 0.01, 0.99);

  // without considering hyperopinions, the expected belief mass distribution differs from the example in [1]
  // namely, the belief mass of the hyperopinion (x1,x3) must be projected to x1 and x2.
  // here, the belief mass is equally distributed during projection
  // in this example, the result is equal to the average and weighted fusion
  OpinionNoBase<3, double>::BeliefType expected_belief_distr{ .495, .01, .495 };
  auto cc_fuse_result = op_a.cc_fuse(op_b);

  for (std::size_t idx{ 0 }; idx < 3; ++idx)
  {
    EXPECT_FLOAT_EQ(cc_fuse_result.belief_masses()[idx], expected_belief_distr[idx]);
  }

  op_a.belief_masses() = OpinionNoBase<3, double>::VacuousBeliefDistr();
  op_b.belief_masses() = OpinionNoBase<3, double>::VacuousBeliefDistr();
  expected_belief_distr.fill(0.);

  cc_fuse_result = op_a.cc_fuse(op_b);

  for (std::size_t idx{ 0 }; idx < 3; ++idx)
  {
    EXPECT_FLOAT_EQ(cc_fuse_result.belief_masses()[idx], expected_belief_distr[idx]);
  }
}

TEST(MultinomialOpinionNoBaseTest, JosangExampleAvgFuse)
{
  OpinionNoBase<3, double> op_a(0.99, 0.01, 0.00);
  OpinionNoBase<3, double> op_b(0.00, 0.01, 0.99);

  OpinionNoBase<3, double>::BeliefType expected_belief_distr{ .495, .01, .495 };
  auto avg_fuse_result = op_a.average_fuse(op_b);

  for (std::size_t idx{ 0 }; idx < 3; ++idx)
  {
    EXPECT_FLOAT_EQ(avg_fuse_result.belief_masses()[idx], expected_belief_distr[idx]);
  }

  op_a.belief_masses() = OpinionNoBase<3, double>::VacuousBeliefDistr();
  op_b.belief_masses() = OpinionNoBase<3, double>::VacuousBeliefDistr();
  expected_belief_distr.fill(0.);

  avg_fuse_result = op_a.average_fuse(op_b);

  for (std::size_t idx{ 0 }; idx < 3; ++idx)
  {
    EXPECT_FLOAT_EQ(avg_fuse_result.belief_masses()[idx], expected_belief_distr[idx]);
  }
}

TEST(MultinomialOpinionNoBaseTest, JosangExampleWeightedFuse)
{
  OpinionNoBase<3, double> op_a(0.99, 0.01, 0.00);
  OpinionNoBase<3, double> op_b(0.00, 0.01, 0.99);

  OpinionNoBase<3, double>::BeliefType expected_belief_distr{ .495, .01, .495 };
  auto weighted_fuse_result = op_a.wb_fuse(op_b);

  for (std::size_t idx{ 0 }; idx < 3; ++idx)
  {
    EXPECT_FLOAT_EQ(weighted_fuse_result.belief_masses()[idx], expected_belief_distr[idx]);
  }

  op_a.belief_masses() = OpinionNoBase<3, double>::VacuousBeliefDistr();
  op_b.belief_masses() = OpinionNoBase<3, double>::VacuousBeliefDistr();
  expected_belief_distr.fill(0.);
  weighted_fuse_result = op_a.wb_fuse(op_b);

  for (std::size_t idx{ 0 }; idx < 3; ++idx)
  {
    EXPECT_FLOAT_EQ(weighted_fuse_result.belief_masses()[idx], expected_belief_distr[idx]);
  }

  op_a = OpinionNoBase<3, double>{ 0.98, 0.01, 0.00 };
  op_b = OpinionNoBase<3, double>{ 0.0, 0.01, 0.90 };

  expected_belief_distr = OpinionNoBase<3, double>::BeliefType{ .889, .01, .083 };
  weighted_fuse_result = op_a.wb_fuse(op_b);

  for (std::size_t idx{ 0 }; idx < 3; ++idx)
  {
    // given numbers in the example are rounded to three digits, respective deviations are expected
    EXPECT_NEAR(weighted_fuse_result.belief_masses()[idx], expected_belief_distr[idx], 0.0005);
  }
}

using TestTypes = ::testing::
    Types<OpinionNoBase<3, float>, OpinionNoBase<6, float>, OpinionNoBase<3, double>, OpinionNoBase<6, double> >;

template <typename OpinionT>
class MultinomialOpinionNoBaseTest : public ::testing::Test
{
  using FloatT = typename OpinionT::FLOAT_t;
  static constexpr std::size_t N = OpinionT::SIZE;

protected:
  void SetUp() override
  {
    variable_ = OpinionT{ OpinionT::NeutralBeliefDistr() }.trust_discount(1 - kUncertaintyMass_);
  }

  static constexpr FloatT kUncertaintyMass_{ 0.3 };
  static constexpr FloatT kEqualBeliefMass_{ (1.0 - kUncertaintyMass_) / N };

  OpinionT variable_;
  OpinionT::BeliefType prior_{ OpinionT::NeutralBeliefDistr() };
};

TYPED_TEST_SUITE(MultinomialOpinionNoBaseTest, TestTypes);

TYPED_TEST(MultinomialOpinionNoBaseTest, Ctor)
{
  for (auto const& value : this->variable_.belief_masses())
  {
    EXPECT_FLOAT_EQ(value, this->kEqualBeliefMass_);
  }
}

TYPED_TEST(MultinomialOpinionNoBaseTest, Uncertainty)
{
  ASSERT_FLOAT_EQ(this->variable_.uncertainty(), this->kUncertaintyMass_);
}

TYPED_TEST(MultinomialOpinionNoBaseTest, GetProjection)
{
  typename TypeParam::FLOAT_t expected_projection =
      this->variable_.belief_masses()[0] + this->kUncertaintyMass_ * this->prior_[0];

  auto projection = this->variable_.getProjection(this->prior_);

  for (auto const& value : projection)
  {
    EXPECT_FLOAT_EQ(value, expected_projection);
  }
}

TYPED_TEST(MultinomialOpinionNoBaseTest, Interpolate)
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

TYPED_TEST(MultinomialOpinionNoBaseTest, DegreeOfConflict)
{
  using Float = typename TypeParam::FLOAT_t;

  TypeParam first{};
  TypeParam second = first;

  EXPECT_FLOAT_EQ(first.degree_of_conflict(second, TypeParam::NeutralBeliefDistr(), TypeParam::NeutralBeliefDistr()),
                  0.F);

  first.belief_masses()[0] = 1;
  second.belief_masses()[1] = 1;
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second, TypeParam::NeutralBeliefDistr(), TypeParam::NeutralBeliefDistr()),
                  1.F);

  first.belief_masses()[0] = 0.5;
  second.belief_masses()[1] = 0.5;
  Float expected_doc = (0.5 + 0.5) / 2. * ((1 - 0.5) * (1 - 0.5));
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second, TypeParam::NeutralBeliefDistr(), TypeParam::NeutralBeliefDistr()),
                  expected_doc);
}

TYPED_TEST(MultinomialOpinionNoBaseTest, CumulativeFusion)
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

  TypeParam var2{ var2_distr };
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

TYPED_TEST(MultinomialOpinionNoBaseTest, BeliefFusion)
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

  TypeParam var2{ var2_distr };
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

TYPED_TEST(MultinomialOpinionNoBaseTest, TrustDiscount)
{
  constexpr typename TypeParam::FLOAT_t kDiscountProp{ 0.8 };
  constexpr typename TypeParam::FLOAT_t kDiscountUncert{ 0.5 };
  constexpr typename TypeParam::FLOAT_t kDiscountPrior{ kDiscountProp };
  OpinionNoBase<2, typename TypeParam::FLOAT_t> kDiscountOpinion{ kDiscountProp * kDiscountUncert,
                                                                  (1 - kDiscountProp) * kDiscountUncert };

  auto var = this->variable_.trust_discount(kDiscountProp);
  auto var2 = this->variable_.trust_discount(kDiscountOpinion, kDiscountPrior);

  EXPECT_FLOAT_EQ(var.uncertainty(), var2.uncertainty());

  typename TypeParam::FLOAT_t certainty = 1 - this->variable_.uncertainty();
  typename TypeParam::FLOAT_t expected_belief_sum = kDiscountProp * certainty;
  auto belief_sum =
      std::accumulate(var.belief_masses().begin(), var.belief_masses().end(), typename TypeParam::FLOAT_t{});
  EXPECT_FLOAT_EQ(belief_sum, expected_belief_sum);
  typename TypeParam::FLOAT_t expected_uncert = 1 - (kDiscountProp * certainty);
  EXPECT_FLOAT_EQ(var.uncertainty(), expected_uncert);
}

TYPED_TEST(MultinomialOpinionNoBaseTest, ReducedOpinions)
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
  }
}

TYPED_TEST(MultinomialOpinionNoBaseTest, StringConversion)
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
