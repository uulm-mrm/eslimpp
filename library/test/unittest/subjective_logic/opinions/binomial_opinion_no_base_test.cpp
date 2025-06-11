#include "gtest/gtest.h"

#include "subjective_logic_lib/opinions/opinion_no_base.hpp"

namespace subjective_logic
{

TEST(BinomialOpinionNoBaseTest, JosangExampleDeductionDetail)
{
  OpinionNoBase<2, double> op_x(.0, .4);
  double x_prior = .5;
  OpinionNoBase<2, double> op_yx(.55, .3);
  OpinionNoBase<2, double> op_ynx(.1, .75);

  auto op_deduced = op_x.deduction(x_prior, op_yx, op_ynx);

  OpinionNoBase<2, double>::BeliefType expected_belief_distr{ 0.15, 0.48 };
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    // example values are rounded to 2 digits, thus the result might have "large" deviations
    EXPECT_NEAR(op_deduced.belief_masses()[idx], expected_belief_distr[idx], 0.005);
  }
}

TEST(BinomialOpinionNoBaseTest, JosangExampleDeduction)
{
  OpinionNoBase<2, double> op_x(.0, .0);
  double x_prior = .8;
  OpinionNoBase<2, double> op_yx(.4, .5);
  OpinionNoBase<2, double> op_ynx(.0, .4);

  auto op_deduced = op_x.deduction(x_prior, op_yx, op_ynx);

  OpinionNoBase<2, double>::BeliefType expected_belief_distr{ 0.26666666666, 0.40 };
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_FLOAT_EQ(op_deduced.belief_masses()[idx], expected_belief_distr[idx]);
  }
}

TEST(BinomialOpinionNoBaseTest, JosangExampleMultiplication)
{
  OpinionNoBase<2, double> op_x(.75, .15);
  double x_prior = .5;
  OpinionNoBase<2, double> op_y(.1, .0);
  double y_prior = 0.2;

  auto op_multi = op_x.multiply(op_y, x_prior, y_prior);

  OpinionNoBase<2, double>::BeliefType expected_belief_distr{ 0.15, 0.15 };
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_NEAR(op_multi.belief_masses()[idx], expected_belief_distr[idx], 0.005);
  }
}

TEST(BinomialOpinionNoBaseTest, JosangExampleComultiplication)
{
  OpinionNoBase<2, double> op_x(.75, .15);
  double x_prior = .5;
  OpinionNoBase<2, double> op_y(.35, .0);
  double y_prior = 0.2;

  auto op_multi = op_x.comultiply(op_y, x_prior, y_prior);

  OpinionNoBase<2, double>::BeliefType expected_belief_distr{ 0.84, 0.06 };
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_NEAR(op_multi.belief_masses()[idx], expected_belief_distr[idx], 0.0055);
  }
}

TEST(BinomialOpinionNoBaseTest, JosangExampleCumulativeUnfusion)
{
  OpinionNoBase<2, double> op_c(.9, .05);
  OpinionNoBase<2, double> op_b(.7, .1);

  auto op_unfused = op_c.cum_unfuse(op_b);

  OpinionNoBase<2, double>::BeliefType expected_belief_distr{ 0.91, 0.03 };
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_NEAR(op_unfused.belief_masses()[idx], expected_belief_distr[idx], 0.005);
  }
}

// TEST(BinomialOpinionNoBaseTest, JosangExampleAverageUnfusion)
// {
// missing in [1]
// OpinionNoBase<2, double> op_c(.9, .05);
// OpinionNoBase<2, double> op_b(.7, .1);
//
// auto op_unfused = op_c.cum_fuse(op_b);
//
// OpinionNoBase<2, double>::BeliefType expected_belief_distr{ 0.91, 0.03 };
// for (std::size_t idx{ 0 }; idx < 2; ++idx)
// {
//   EXPECT_NEAR(op_unfused.belief_masses()[idx], expected_belief_distr[idx], 0.005);
// }
// }

using TestTypes = ::testing::Types<float, double>;

template <typename T>
using OpinionT = OpinionNoBase<2, T>;

template <typename T>
class BinomialOpinionNoBaseTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    variable_ = OpinionT<T>(belief_masses_);
  }

  OpinionT<T> variable_;
  OpinionT<T>::BeliefType belief_masses_{ 0.7, 0.1 };
  T uncertainty = 1 - belief_masses_[0] - belief_masses_[1];
  T prior_ = 0.5;
};

TYPED_TEST_SUITE(BinomialOpinionNoBaseTest, TestTypes);

TYPED_TEST(BinomialOpinionNoBaseTest, Ctor)
{
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
}

TYPED_TEST(BinomialOpinionNoBaseTest, IsValid)
{
  EXPECT_TRUE(this->variable_.is_valid());

  auto exceed = this->variable_;
  exceed.belief_masses()[0] += 2;
  EXPECT_FALSE(exceed.is_valid());

  auto negative = this->variable_;
  negative.belief_masses()[0] = -0.1;
  EXPECT_FALSE(negative.is_valid());
}

TYPED_TEST(BinomialOpinionNoBaseTest, Complement)
{
  OpinionT<TypeParam> test_var = this->variable_.complement();

  EXPECT_FLOAT_EQ(test_var.belief(), this->variable_.disbelief());
  EXPECT_FLOAT_EQ(test_var.disbelief(), this->variable_.belief());
}

TYPED_TEST(BinomialOpinionNoBaseTest, Uncertainty)
{
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);
}

TYPED_TEST(BinomialOpinionNoBaseTest, BeliefDisbeliefAccessors)
{
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(reinterpret_cast<const OpinionT<TypeParam>&>(this->variable_).belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(reinterpret_cast<const OpinionT<TypeParam>&>(this->variable_).disbelief(), this->belief_masses_[1]);
}

TYPED_TEST(BinomialOpinionNoBaseTest, VacuouseBeliefDistr)
{
  typename OpinionT<TypeParam>::BeliefType test = OpinionT<TypeParam>::VacuousBeliefDistr();
  typename OpinionT<TypeParam>::BeliefType expected{ 0.0, 0.0 };

  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_FLOAT_EQ(test[idx], expected[idx]);
  }
}

TYPED_TEST(BinomialOpinionNoBaseTest, VacuouseBeliefOpinion)
{
  OpinionT<TypeParam> test = OpinionT<TypeParam>::VacuousBeliefOpinion();
  OpinionT<TypeParam> expected{ 0.0, 0.0 };

  EXPECT_EQ(test, expected);
}

TYPED_TEST(BinomialOpinionNoBaseTest, NeutralBeliefDistr)
{
  typename OpinionT<TypeParam>::BeliefType test = OpinionT<TypeParam>::NeutralBeliefDistr();
  typename OpinionT<TypeParam>::BeliefType expected{ 0.5, 0.5 };

  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_FLOAT_EQ(test[idx], expected[idx]);
  }
}

TYPED_TEST(BinomialOpinionNoBaseTest, NeutralBeliefOpinion)
{
  OpinionT<TypeParam> test = OpinionT<TypeParam>::NeutralBeliefOpinion();
  OpinionT<TypeParam> expected{ 0.5, 0.5 };

  EXPECT_EQ(test, expected);
}

TYPED_TEST(BinomialOpinionNoBaseTest, DissonanceTest)
{
  OpinionT<TypeParam> test = OpinionT<TypeParam>::NeutralBeliefOpinion();
  TypeParam expected{ 1.0 };

  EXPECT_EQ(test.dissonance(), expected);

  test.belief_masses()[0] = 1.0;
  test.belief_masses()[1] = 0.0;
  expected = 0.0;
  EXPECT_EQ(test.dissonance(), expected);

  test.belief_masses()[0] = 0.0;
  test.belief_masses()[1] = 1.0;
  expected = 0.0;
  EXPECT_EQ(test.dissonance(), expected);
}

TYPED_TEST(BinomialOpinionNoBaseTest, GetProjection)
{
  TypeParam expected_projection = this->belief_masses_[0] + this->uncertainty * this->prior_;

  EXPECT_FLOAT_EQ(this->variable_.getBinomialProjection(this->prior_), expected_projection);
}

TYPED_TEST(BinomialOpinionNoBaseTest, GetProbability)
{
  TypeParam total_mass = (1 - this->uncertainty);
  TypeParam expected_probability = this->belief_masses_[0] / total_mass;

  EXPECT_FLOAT_EQ(this->variable_.getProbability(), expected_probability);
}

TYPED_TEST(BinomialOpinionNoBaseTest, GetProbabilities)
{
  TypeParam total_mass = (1 - this->uncertainty);
  TypeParam expected_probability = this->belief_masses_[0] / total_mass;

  auto probs = this->variable_.getProbabilities();
  // error is near enough but higher than float equal
  EXPECT_NEAR(probs[0], expected_probability, 1e-6);
  EXPECT_NEAR(probs[1], 1. - expected_probability, 1e-6);
}

TYPED_TEST(BinomialOpinionNoBaseTest, DegreeOfConflict)
{
  // all priors are set to 0.5 by default

  OpinionT<TypeParam> first{ 0, 0 };
  OpinionT<TypeParam> second = first;
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), 0.F);

  first = OpinionT<TypeParam>{ 1., 0. };
  second = OpinionT<TypeParam>{ 0., 1. };
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), 1.F);

  first = OpinionT<TypeParam>{ 0.5, 0. };
  second = OpinionT<TypeParam>{ 0., 0.5 };
  TypeParam expected_doc = (0.5 + 0.5) / 2. * ((1 - 0.5) * (1 - 0.5));
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), expected_doc);
}

TYPED_TEST(BinomialOpinionNoBaseTest, UncertaintyDifferential)
{
  constexpr TypeParam second_belief{ 0.3 };
  constexpr TypeParam second_disbelief{ 0.2 };
  OpinionT<TypeParam> second_op{ second_belief, second_disbelief };

  TypeParam uncert_diff_first = this->variable_.uncertainty_differential(second_op);
  TypeParam uncert_diff_second = second_op.uncertainty_differential(this->variable_);

  TypeParam total_uncertainty = this->variable_.uncertainty() + second_op.uncertainty();
  TypeParam expected_first = this->variable_.uncertainty() / total_uncertainty;
  TypeParam expected_second = second_op.uncertainty() / total_uncertainty;

  EXPECT_FLOAT_EQ(uncert_diff_first, expected_first);
  EXPECT_FLOAT_EQ(uncert_diff_second, expected_second);
}

TYPED_TEST(BinomialOpinionNoBaseTest, TrustRevision)
{
  constexpr TypeParam second_belief{ 0.3 };
  constexpr TypeParam second_disbelief{ 0.2 };
  OpinionT<TypeParam> second_op{ second_belief, second_disbelief };

  constexpr TypeParam conflict = 0.4;

  TypeParam revision_factor = second_op.uncertainty_differential(this->variable_) * conflict;
  TypeParam expected_belief_mass = second_op.belief() * (1 - revision_factor);
  TypeParam expected_uncertainty_mass = second_op.uncertainty() * (1 - revision_factor);

  OpinionT<TypeParam> revised_trust = second_op.revise_trust(conflict, this->variable_);
  auto const_revision_access =
      reinterpret_cast<const OpinionT<TypeParam>&>(second_op).revise_trust(conflict, this->variable_);
  auto third_op = second_op.revise_trust(revision_factor);
  second_op.revise_trust_(conflict, this->variable_);

  EXPECT_EQ(revised_trust, second_op);
  EXPECT_EQ(const_revision_access, second_op);
  EXPECT_EQ(third_op, second_op);

  EXPECT_FLOAT_EQ(revised_trust.uncertainty(), expected_uncertainty_mass);
  EXPECT_FLOAT_EQ(revised_trust.belief(), expected_belief_mass);
}

TYPED_TEST(BinomialOpinionNoBaseTest, CumulativeFusion)
{
  OpinionT<TypeParam> neural_element{ 0, 0 };

  this->variable_.cum_fuse_(neural_element);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);

  OpinionT<TypeParam> var2{ 0.2, 0.3 };
  auto fused = this->variable_.cum_fuse(var2);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);

  TypeParam expected_belief = 0.65;
  TypeParam expected_uncertainty = 0.16666666;
  EXPECT_FLOAT_EQ(fused.belief(), expected_belief);
  EXPECT_FLOAT_EQ(fused.uncertainty(), expected_uncertainty);
}

TYPED_TEST(BinomialOpinionNoBaseTest, CumulativeUnfusion)
{
  OpinionT<TypeParam> var{ 0.2, 0.3 };

  auto cum_fused = this->variable_.cum_fuse(var);
  auto unfused = cum_fused.cum_unfuse(var);

  EXPECT_FLOAT_EQ(this->variable_.belief(), unfused.belief());
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), unfused.disbelief());
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), unfused.uncertainty());
}

TYPED_TEST(BinomialOpinionNoBaseTest, Harmony)
{
  OpinionT<TypeParam> var2{ 0.2, 0.3 };
  auto harmony = this->variable_.harmony(var2);
  EXPECT_FLOAT_EQ(harmony[0], 0.5299999);
}

TYPED_TEST(BinomialOpinionNoBaseTest, Conflict)
{
  OpinionT<TypeParam> var2{ 0.2, 0.3 };
  auto conflict = this->variable_.conflict(var2);
  TypeParam expected_conflict =
      this->variable_.belief() * var2.disbelief() + var2.belief() * this->variable_.disbelief();
  EXPECT_FLOAT_EQ(conflict, expected_conflict);
}

TYPED_TEST(BinomialOpinionNoBaseTest, BeliefFusion)
{
  OpinionT<TypeParam> neural_element{ 0, 0 };

  this->variable_.bc_fuse_(neural_element);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);

  OpinionT<TypeParam> var2{ 0.2, 0.3 };
  auto fused = this->variable_.bc_fuse(var2);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);

  typename OpinionT<TypeParam>::BeliefType harmony = this->variable_.harmony(var2);
  TypeParam conflict = this->variable_.conflict(var2);
  TypeParam denom = 1 - conflict;

  TypeParam expected_belief = harmony[0] / denom;
  TypeParam expected_uncertainty = this->variable_.uncertainty() * var2.uncertainty() / denom;
  EXPECT_FLOAT_EQ(fused.belief(), expected_belief);
  EXPECT_FLOAT_EQ(fused.uncertainty(), expected_uncertainty);

  OpinionT<TypeParam> conflict_1{ 1., 0. };
  OpinionT<TypeParam> conflict_2{ 0., 1. };
  auto expected_belief_conflict = OpinionT<TypeParam>::NeutralBeliefDistr();
  auto test_bc_fuse_conflict = conflict_1.bc_fuse(conflict_2);
  EXPECT_FLOAT_EQ(test_bc_fuse_conflict.belief(), expected_belief_conflict[0]);
  EXPECT_FLOAT_EQ(test_bc_fuse_conflict.disbelief(), expected_belief_conflict[1]);
}

TYPED_TEST(BinomialOpinionNoBaseTest, TrustDiscount)
{
  constexpr TypeParam kDiscountProp{ 0.8 };
  constexpr TypeParam kDiscountUncert{ 0.5 };
  constexpr TypeParam kDiscountPrior{ kDiscountProp };
  OpinionT<TypeParam> kDiscountOpinion{ kDiscountProp * kDiscountUncert, (1 - kDiscountProp) * kDiscountUncert };

  auto var = this->variable_.trust_discount(kDiscountProp);
  auto var2 = this->variable_.trust_discount(kDiscountOpinion, kDiscountPrior);

  EXPECT_FLOAT_EQ(var.belief(), var2.belief());
  EXPECT_FLOAT_EQ(var.uncertainty(), var2.uncertainty());

  TypeParam expected_belief = kDiscountProp * this->variable_.belief();
  EXPECT_FLOAT_EQ(var.belief(), expected_belief);
  TypeParam certainty = 1 - this->variable_.uncertainty();
  TypeParam expected_uncert = 1 - (kDiscountProp * certainty);
  EXPECT_FLOAT_EQ(var.uncertainty(), expected_uncert);
}

TYPED_TEST(BinomialOpinionNoBaseTest, LimitedTrustDiscount)
{
  OpinionT<TypeParam> test_op{ 1.0, 0. };
  TypeParam uncert_limit{ 0.3 };

  auto discounted_op = test_op.limited_trust_discount(uncert_limit, .1);
  EXPECT_FLOAT_EQ(discounted_op.uncertainty(), uncert_limit);

  OpinionNoBase<2, TypeParam> trust{ 0.5, 0.5 };
  constexpr TypeParam prior{ 0.8 };
  TypeParam expected_prop{ trust.getBinomialProjection(prior) };

  discounted_op =
      reinterpret_cast<const OpinionT<TypeParam>&>(test_op).limited_trust_discount(uncert_limit, trust, prior);
  auto discounted_op2 =
      reinterpret_cast<const OpinionT<TypeParam>&>(test_op).limited_trust_discount(uncert_limit, expected_prop);

  EXPECT_FLOAT_EQ(discounted_op.belief(), discounted_op2.belief());
  EXPECT_FLOAT_EQ(discounted_op.disbelief(), discounted_op2.disbelief());

  uncert_limit = 0.99;
  discounted_op = test_op.limited_trust_discount(uncert_limit, trust, prior);
  discounted_op2 = test_op.limited_trust_discount(uncert_limit, expected_prop);

  EXPECT_FLOAT_EQ(discounted_op.belief(), discounted_op2.belief());
  EXPECT_FLOAT_EQ(discounted_op.disbelief(), discounted_op2.disbelief());
}

TYPED_TEST(BinomialOpinionNoBaseTest, DeductionZeroDenom)
{
  // issue occured in this case, since it resulted in nan values...

  OpinionT<TypeParam> var2{ 0.2, 0.3 };
  TypeParam base_x{ 1. };
  OpinionT<TypeParam> cond_1{ 0.0, 0.0 };
  OpinionT<TypeParam> cond_2{ 0.5, 0.5 };

  OpinionT<TypeParam> result = var2.deduction(base_x, cond_1, cond_2);

  // Nan Check
  EXPECT_FALSE(result.belief() != result.belief());
  EXPECT_FALSE(result.disbelief() != result.disbelief());
}

TYPED_TEST(BinomialOpinionNoBaseTest, StringConversion)
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
