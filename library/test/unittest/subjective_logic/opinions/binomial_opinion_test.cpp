#include "gtest/gtest.h"

#include "subjective_logic_lib/opinions/opinion.hpp"

namespace subjective_logic
{

TEST(BinomialOpinionTest, JosangExampleDeductionDetail)
{
  Opinion<2, double> op_x(.0, .4, .5);
  Opinion<2, double> op_yx(.55, .3);
  Opinion<2, double> op_ynx(.1, .75);

  auto op_deduced = op_x.deduction(op_yx, op_ynx);

  Opinion<2, double>::BeliefType expected_belief_distr{ 0.15, 0.48 };
  double expected_prior = 0.38;
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    // example values are rounded to 2 digits, thus the result might have "large" deviations
    EXPECT_NEAR(op_deduced.belief_masses()[idx], expected_belief_distr[idx], 0.005);
  }
  EXPECT_NEAR(op_deduced.getBinomialPrior(), expected_prior, 0.005);
}

TEST(BinomialOpinionTest, JosangExampleMultiplication)
{
  Opinion<2, double> op_x(0.75, 0.15, 0.5);
  Opinion<2, double> op_y(0.1, 0.0, 0.2);

  auto op_multi = op_x.multiply(op_y);

  Opinion<2, double>::BeliefType expected_belief_distr{ 0.15, 0.15 };
  double expected_prior = 0.1;
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    // example values are rounded to 2 digits, thus the result might have "large" deviations
    EXPECT_NEAR(op_multi.belief_masses()[idx], expected_belief_distr[idx], 0.005);
  }
  EXPECT_NEAR(op_multi.getBinomialPrior(), expected_prior, 0.005);
}

TEST(BinomialOpinionTest, JosangExampleComultiplication)
{
  Opinion<2, double> op_x(0.75, 0.15, 0.5);
  Opinion<2, double> op_y(0.35, 0.0, 0.2);

  auto op_multi = op_x.comultiply(op_y);

  Opinion<2, double>::BeliefType expected_belief_distr{ 0.84, 0.06 };
  double expected_prior = 0.6;
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    // example values are rounded to 2 digits, thus the result might have "large" deviations
    EXPECT_NEAR(op_multi.belief_masses()[idx], expected_belief_distr[idx], 0.0055);
  }
  EXPECT_NEAR(op_multi.getBinomialPrior(), expected_prior, 0.0055);
}

TEST(BinomialOpinionTest, JosangExampleCumulativeUnfusion)
{
  Opinion<2, double> op_c(.9, .05, 0.5);
  Opinion<2, double> op_b(.7, .1, 0.5);

  auto op_unfused = op_c.cum_unfuse(op_b);

  OpinionNoBase<2, double>::BeliefType expected_belief_distr{ 0.91, 0.03 };
  auto expected_prior = op_c.getBinomialPrior();
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_NEAR(op_unfused.belief_masses()[idx], expected_belief_distr[idx], 0.005);
  }
  EXPECT_NEAR(op_unfused.getBinomialPrior(), expected_prior, 0.005);
}

using TestTypes = ::testing::Types<Opinion<2, float>, Opinion<2, double>>;

template <typename OpinionT>
class BinomialOpinionTest : public ::testing::Test
{
  using FloatT = typename OpinionT::FLOAT_t;
  static constexpr std::size_t N = OpinionT::SIZE;

protected:
  void SetUp() override
  {
    variable_ = OpinionT(belief_masses_, prior_);
  }

  OpinionT variable_;
  OpinionT::BeliefType belief_masses_{ 0.7, 0.1 };
  FloatT uncertainty = 1 - belief_masses_[0] - belief_masses_[1];
  OpinionT::BeliefType prior_{ 0.5, 0.5 };
};

TYPED_TEST_SUITE(BinomialOpinionTest, TestTypes);

TYPED_TEST(BinomialOpinionTest, Ctor)
{
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);

  typename TypeParam::BeliefType prior = this->variable_.prior_belief_masses();
  EXPECT_FLOAT_EQ(prior[0], this->prior_[0]);
  EXPECT_FLOAT_EQ(prior[1], this->prior_[1]);

  typename TypeParam::FLOAT_t kbelief_mass{ 0.2 };
  typename TypeParam::FLOAT_t kdisbelief_mass{ 0.3 };
  TypeParam opinion{ kbelief_mass, kdisbelief_mass };

  typename TypeParam::BeliefType prior2 = opinion.prior_belief_masses();
  EXPECT_FLOAT_EQ(opinion.belief(), kbelief_mass);
  EXPECT_FLOAT_EQ(opinion.disbelief(), kdisbelief_mass);
  EXPECT_FLOAT_EQ(opinion.prior_belief(), 0.5);
  EXPECT_FLOAT_EQ(opinion.prior_disbelief(), 0.5);
  EXPECT_FLOAT_EQ(prior2[0], 0.5);
  EXPECT_FLOAT_EQ(prior2[1], 0.5);
}

TYPED_TEST(BinomialOpinionTest, IsValid)
{
  EXPECT_TRUE(this->variable_.is_valid());

  auto exceed = this->variable_;
  exceed.prior_belief_masses()[0] += 2;
  EXPECT_FALSE(exceed.is_valid());

  auto negative = this->variable_;
  negative.prior_belief_masses()[0] = -0.1;
  EXPECT_FALSE(negative.is_valid());
}

TYPED_TEST(BinomialOpinionTest, Complement)
{
  TypeParam test_var = this->variable_.complement();

  EXPECT_FLOAT_EQ(test_var.belief(), this->variable_.disbelief());
  EXPECT_FLOAT_EQ(test_var.disbelief(), this->variable_.belief());
}

TYPED_TEST(BinomialOpinionTest, Uncertainty)
{
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);
}

TYPED_TEST(BinomialOpinionTest, NeutralBeliefDistr)
{
  // for reasons, gcovr does not recognize any of these calls
  typename TypeParam::BeliefType test = TypeParam::NeutralBeliefDistr();
  typename TypeParam::BeliefType expected{ 0.5, 0.5 };

  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_FLOAT_EQ(test[idx], expected[idx]);
  }

  TypeParam test_op{};
  test = test_op.NeutralBeliefDistr();
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_FLOAT_EQ(test[idx], expected[idx]);
  }
}

TYPED_TEST(BinomialOpinionTest, VacuousBeliefDistr)
{
  typename TypeParam::BeliefType test = TypeParam::VacuousBeliefDistr();
  typename TypeParam::BeliefType expected{ 0.0, 0.0 };

  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_FLOAT_EQ(test[idx], expected[idx]);
  }
}

TYPED_TEST(BinomialOpinionTest, NeutralBeliefOpinion)
{
  TypeParam test = TypeParam::NeutralBeliefOpinion();
  TypeParam expected{ TypeParam::NeutralBeliefDistr(), TypeParam::NeutralBeliefDistr() };

  EXPECT_EQ(test, expected);
}

TYPED_TEST(BinomialOpinionTest, VacuousBeliefOpinion)
{
  TypeParam test = TypeParam::VacuousBeliefOpinion();
  TypeParam expected{ TypeParam::VacuousBeliefDistr(), TypeParam::VacuousBeliefDistr() };

  EXPECT_EQ(test, expected);
}

TYPED_TEST(BinomialOpinionTest, DogmaticTrust)
{
  // for reasons, gcovr does not recognize any of these calls
  TypeParam trust = TypeParam::DogmaticTrust();
  EXPECT_FLOAT_EQ(trust.belief(), 1.0);
  EXPECT_FLOAT_EQ(trust.disbelief(), 0.0);
  EXPECT_FLOAT_EQ(trust.uncertainty(), 0.0);

  TypeParam test_op{};
  trust = test_op.DogmaticTrust();
  EXPECT_FLOAT_EQ(trust.belief(), 1.0);
  EXPECT_FLOAT_EQ(trust.disbelief(), 0.0);
  EXPECT_FLOAT_EQ(trust.uncertainty(), 0.0);
}

TYPED_TEST(BinomialOpinionTest, BeliefMassesAccessor)
{
  auto& belief_mass_ref = this->variable_.belief_masses();
  EXPECT_EQ(&belief_mass_ref, &this->variable_.as_no_base().belief_masses());

  auto const& belief_mass_ref_2 = reinterpret_cast<const TypeParam&>(this->variable_).belief_masses();
  EXPECT_EQ(&belief_mass_ref_2, &this->variable_.as_no_base().belief_masses());
}

TYPED_TEST(BinomialOpinionTest, PriorAccessor)
{
  EXPECT_FLOAT_EQ(this->variable_.prior_belief(), this->prior_[0]);
  EXPECT_FLOAT_EQ(this->variable_.prior_disbelief(), this->prior_[1]);
  EXPECT_FLOAT_EQ(reinterpret_cast<const TypeParam&>(this->variable_).prior_belief(), this->prior_[0]);
  EXPECT_FLOAT_EQ(reinterpret_cast<const TypeParam&>(this->variable_).prior_disbelief(), this->prior_[1]);

  auto& prior_ref = this->variable_.prior_belief_masses();
  auto const& prior_ref_2 = reinterpret_cast<const TypeParam&>(this->variable_).prior_belief_masses();

  EXPECT_EQ(&prior_ref, &this->variable_.prior_belief_masses());
  EXPECT_EQ(&prior_ref_2, &this->variable_.prior_belief_masses());
}

TYPED_TEST(BinomialOpinionTest, GetProjection)
{
  typename TypeParam::FLOAT_t expected_projection = this->belief_masses_[0] + this->uncertainty * this->prior_[0];

  EXPECT_FLOAT_EQ(this->variable_.getBinomialProjection(), expected_projection);
  EXPECT_FLOAT_EQ(this->variable_.getProjection()[0], expected_projection);
}

TYPED_TEST(BinomialOpinionTest, GetProbability)
{
  using Float = typename TypeParam::FLOAT_t;
  Float total_mass = (1 - this->uncertainty);
  Float expected_probability = this->belief_masses_[0] / total_mass;

  EXPECT_FLOAT_EQ(this->variable_.getProbability(), expected_probability);

  auto probs = this->variable_.getProbabilities();
  EXPECT_FLOAT_EQ(probs[0], expected_probability);
  EXPECT_NEAR(probs[1], 1 - expected_probability, 1e-6);
}

TYPED_TEST(BinomialOpinionTest, DegreeOfConflict)
{
  using Float = typename TypeParam::FLOAT_t;

  TypeParam first{ 0, 0, 0.5 };
  TypeParam second = first;
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), 0.F);

  first = TypeParam{ 1., 0., 0.5 };
  second = TypeParam{ 0., 1., 0.5 };
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), 1.F);

  first = TypeParam{ 0.5, 0., 0.5 };
  second = TypeParam{ 0., 0.5, 0.5 };
  Float expected_doc = (0.5 + 0.5) / 2. * ((1 - 0.5) * (1 - 0.5));
  EXPECT_FLOAT_EQ(first.degree_of_conflict(second), expected_doc);
}

TYPED_TEST(BinomialOpinionTest, UncertaintyDifferential)
{
  using Float = typename TypeParam::FLOAT_t;
  constexpr Float second_belief{ 0.3 };
  constexpr Float second_disbelief{ 0.2 };
  TypeParam second_op{ second_belief, second_disbelief, 0.5 };

  Float uncert_diff_first = this->variable_.uncertainty_differential(second_op);
  Float uncert_diff_second = second_op.uncertainty_differential(this->variable_);

  Float total_uncertainty = this->variable_.uncertainty() + second_op.uncertainty();
  Float expected_first = this->variable_.uncertainty() / total_uncertainty;
  Float expected_second = second_op.uncertainty() / total_uncertainty;

  EXPECT_FLOAT_EQ(uncert_diff_first, expected_first);
  EXPECT_FLOAT_EQ(uncert_diff_second, expected_second);
}

TYPED_TEST(BinomialOpinionTest, TrustRevision)
{
  using Float = typename TypeParam::FLOAT_t;
  constexpr Float second_belief{ 0.3 };
  constexpr Float second_disbelief{ 0.2 };
  TypeParam second_op{ second_belief, second_disbelief, 0.5 };

  constexpr Float conflict = 0.4;

  Float revision_factor = second_op.uncertainty_differential(this->variable_) * conflict;
  Float expected_belief_mass = second_op.belief() * (1 - revision_factor);
  Float expected_uncertainty_mass = second_op.uncertainty() * (1 - revision_factor);

  TypeParam revised_trust = second_op.revise_trust(conflict, this->variable_);
  TypeParam copy_op{ second_op };
  second_op.revise_trust_(conflict, this->variable_);
  TypeParam revised_copy_op = copy_op.revise_trust(revision_factor);

  EXPECT_FLOAT_EQ(revised_trust.belief(), second_op.belief());
  EXPECT_FLOAT_EQ(revised_trust.disbelief(), second_op.disbelief());
  EXPECT_FLOAT_EQ(revised_trust.uncertainty(), second_op.uncertainty());
  EXPECT_FLOAT_EQ(revised_copy_op.belief(), second_op.belief());
  EXPECT_FLOAT_EQ(revised_copy_op.disbelief(), second_op.disbelief());
  EXPECT_FLOAT_EQ(revised_copy_op.uncertainty(), second_op.uncertainty());

  EXPECT_FLOAT_EQ(revised_trust.uncertainty(), expected_uncertainty_mass);
  EXPECT_FLOAT_EQ(revised_trust.belief(), expected_belief_mass);
}

TYPED_TEST(BinomialOpinionTest, CumulativeFusion)
{
  using Float = TypeParam::FLOAT_t;
  TypeParam neutral_element{ 0, 0, this->prior_[0] };

  this->variable_.cum_fuse_(neutral_element);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);
  EXPECT_FLOAT_EQ(this->variable_.prior_belief(), this->prior_[0]);
  EXPECT_FLOAT_EQ(this->variable_.prior_disbelief(), this->prior_[1]);

  TypeParam var2{ 0.2, 0.3, 0.2 };
  auto fused = this->variable_.cum_fuse(var2);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);
  EXPECT_FLOAT_EQ(this->variable_.prior_belief(), this->prior_[0]);
  EXPECT_FLOAT_EQ(this->variable_.prior_disbelief(), this->prior_[1]);

  Float expected_belief = 0.65;
  Float expected_disbelief = 0.18333333;
  Float expected_uncertainty = 0.16666666;

  Float uncert_var = this->variable_.uncertainty();
  Float uncert_var2 = var2.uncertainty();
  Float expected_prior = this->variable_.prior_belief() * uncert_var2 + var2.prior_belief() * uncert_var -
                         (this->variable_.prior_belief() + var2.prior_belief()) * uncert_var * uncert_var2;
  expected_prior /= uncert_var + uncert_var2 - 2. * uncert_var * uncert_var2;
  EXPECT_FLOAT_EQ(fused.belief(), expected_belief);
  EXPECT_FLOAT_EQ(fused.disbelief(), expected_disbelief);
  EXPECT_FLOAT_EQ(fused.uncertainty(), expected_uncertainty);
  EXPECT_FLOAT_EQ(fused.prior_belief(), expected_prior);
  EXPECT_FLOAT_EQ(fused.prior_disbelief(), 1 - expected_prior);

  TypeParam neutral_element2{ neutral_element };
  auto cc_fuse_neutral = neutral_element.cum_fuse(neutral_element2);
  EXPECT_FLOAT_EQ(cc_fuse_neutral.belief(), neutral_element.belief());
  EXPECT_FLOAT_EQ(cc_fuse_neutral.disbelief(), neutral_element.disbelief());
  EXPECT_FLOAT_EQ(cc_fuse_neutral.uncertainty(), neutral_element.uncertainty());
}

TYPED_TEST(BinomialOpinionTest, BeliefFusion)
{
  using Float = TypeParam::FLOAT_t;
  TypeParam neutral_element{ 0, 0, this->prior_[0] };

  this->variable_.bc_fuse_(neutral_element);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);
  EXPECT_FLOAT_EQ(this->variable_.prior_belief(), this->prior_[0]);
  EXPECT_FLOAT_EQ(this->variable_.prior_disbelief(), this->prior_[1]);

  TypeParam var2{ 0.2, 0.3, 0.2 };
  auto fused = this->variable_.bc_fuse(var2);
  EXPECT_FLOAT_EQ(this->variable_.belief(), this->belief_masses_[0]);
  EXPECT_FLOAT_EQ(this->variable_.disbelief(), this->belief_masses_[1]);
  EXPECT_FLOAT_EQ(this->variable_.uncertainty(), this->uncertainty);
  EXPECT_FLOAT_EQ(this->variable_.prior_belief(), this->prior_[0]);
  EXPECT_FLOAT_EQ(this->variable_.prior_disbelief(), this->prior_[1]);

  typename TypeParam::BeliefType harmony = this->variable_.harmony(var2);
  Float conflict = this->variable_.conflict(var2);
  Float denom = 1 - conflict;

  Float expected_belief = harmony[0] / denom;
  Float expected_uncertainty = this->variable_.uncertainty() * var2.uncertainty() / denom;
  Float uncert_var = this->variable_.uncertainty();
  Float uncert_var2 = var2.uncertainty();
  Float expected_prior = this->variable_.prior_belief() * (1. - uncert_var) + var2.prior_belief() * (1. - uncert_var2);
  expected_prior /= 2. - uncert_var - uncert_var2;
  EXPECT_FLOAT_EQ(fused.belief(), expected_belief);
  EXPECT_FLOAT_EQ(fused.uncertainty(), expected_uncertainty);
  EXPECT_FLOAT_EQ(fused.prior_belief(), expected_prior);
  EXPECT_FLOAT_EQ(fused.prior_disbelief(), 1 - expected_prior);

  TypeParam neutral_element2{ neutral_element };
  auto bc_fuse_neutral = neutral_element.bc_fuse(neutral_element2);
  EXPECT_FLOAT_EQ(bc_fuse_neutral.belief(), neutral_element.belief());
  EXPECT_FLOAT_EQ(bc_fuse_neutral.disbelief(), neutral_element.disbelief());
  EXPECT_FLOAT_EQ(bc_fuse_neutral.uncertainty(), neutral_element.uncertainty());
  EXPECT_FLOAT_EQ(bc_fuse_neutral.prior_belief(), neutral_element.prior_belief());
  EXPECT_FLOAT_EQ(bc_fuse_neutral.prior_disbelief(), neutral_element.prior_disbelief());
}

TYPED_TEST(BinomialOpinionTest, AverageFusion)
{
  using Float = TypeParam::FLOAT_t;
  TypeParam neutral_element{ 0, 0, this->prior_[0] };

  TypeParam avg_fused = this->variable_.average_fuse(neutral_element);

  Float expected_prior_bel = this->variable_.prior_belief() * (1 - this->variable_.uncertainty()) +
                             neutral_element.prior_belief() * (1 - neutral_element.uncertainty());
  expected_prior_bel /= 2 - this->variable_.uncertainty() - neutral_element.uncertainty();
  Float expected_prior_dis = this->variable_.prior_disbelief() * (1 - this->variable_.uncertainty()) +
                             neutral_element.prior_disbelief() * (1 - neutral_element.uncertainty());
  expected_prior_dis /= 2 - this->variable_.uncertainty() - neutral_element.uncertainty();

  EXPECT_FLOAT_EQ(avg_fused.prior_belief(), expected_prior_bel);
  EXPECT_FLOAT_EQ(avg_fused.prior_disbelief(), expected_prior_dis);

  TypeParam neutral_element2{ neutral_element };
  auto average_fuse_neutral = neutral_element.average_fuse(neutral_element2);
  EXPECT_FLOAT_EQ(average_fuse_neutral.prior_belief(), neutral_element.prior_belief());
  EXPECT_FLOAT_EQ(average_fuse_neutral.prior_disbelief(), neutral_element.prior_disbelief());
}

TYPED_TEST(BinomialOpinionTest, WeightedFusion)
{
  using Float = TypeParam::FLOAT_t;
  TypeParam neutral_element{ 0, 0, this->prior_[0] };

  TypeParam wb_fuse = this->variable_.wb_fuse(neutral_element);

  Float expected_prior_bel = this->variable_.prior_belief() * (1 - this->variable_.uncertainty()) +
                             neutral_element.prior_belief() * (1 - neutral_element.uncertainty());
  expected_prior_bel /= 2 - this->variable_.uncertainty() - neutral_element.uncertainty();
  Float expected_prior_dis = this->variable_.prior_disbelief() * (1 - this->variable_.uncertainty()) +
                             neutral_element.prior_disbelief() * (1 - neutral_element.uncertainty());
  expected_prior_dis /= 2 - this->variable_.uncertainty() - neutral_element.uncertainty();

  EXPECT_FLOAT_EQ(wb_fuse.prior_belief(), expected_prior_bel);
  EXPECT_FLOAT_EQ(wb_fuse.prior_disbelief(), expected_prior_dis);

  TypeParam neutral_element2{ neutral_element };
  auto wb_fuse_neutral = neutral_element.wb_fuse(neutral_element2);
  EXPECT_FLOAT_EQ(wb_fuse_neutral.prior_belief(), neutral_element.prior_belief());
  EXPECT_FLOAT_EQ(wb_fuse_neutral.prior_disbelief(), neutral_element.prior_disbelief());
}

TYPED_TEST(BinomialOpinionTest, CcFusion)
{
  TypeParam neutral_element{ 0, 0, this->prior_[0] };

  TypeParam cc_fuse = this->variable_.cc_fuse(neutral_element);

  // no update expected
  EXPECT_FLOAT_EQ(cc_fuse.prior_belief(), this->variable_.prior_belief());
  EXPECT_FLOAT_EQ(cc_fuse.prior_disbelief(), this->variable_.prior_disbelief());
}

TYPED_TEST(BinomialOpinionTest, TrustDiscount)
{
  using Float = TypeParam::FLOAT_t;
  constexpr Float kDiscountProp{ 0.8 };
  constexpr Float kDiscountUncert{ 0.5 };
  constexpr Float kDiscountPrior{ kDiscountProp };
  Trust<Float> kDiscountOpinion{ kDiscountProp * kDiscountUncert,
                                 (1 - kDiscountProp) * kDiscountUncert,
                                 kDiscountPrior };

  auto var = this->variable_.trust_discount(kDiscountProp);
  auto var2 = this->variable_.trust_discount(kDiscountOpinion);

  EXPECT_FLOAT_EQ(var.belief(), var2.belief());
  EXPECT_FLOAT_EQ(var.uncertainty(), var2.uncertainty());

  Float expected_belief = kDiscountProp * this->variable_.belief();
  EXPECT_FLOAT_EQ(var.belief(), expected_belief);
  Float certainty = 1 - this->variable_.uncertainty();
  Float expected_uncert = 1 - (kDiscountProp * certainty);
  EXPECT_FLOAT_EQ(var.uncertainty(), expected_uncert);
}

TYPED_TEST(BinomialOpinionTest, LimitedTrustDiscount)
{
  using Float = TypeParam::FLOAT_t;
  constexpr TypeParam vac_trust{ 0., 0., 0.1 };
  constexpr Float discount_factor{ vac_trust.getBinomialProjection() };
  constexpr Float limit{ 0.5 };

  auto var = this->variable_.limited_trust_discount(limit, vac_trust);
  auto var2 = this->variable_.limited_trust_discount(limit, discount_factor);

  EXPECT_FLOAT_EQ(var.belief(), var2.belief());
  EXPECT_FLOAT_EQ(var.uncertainty(), var2.uncertainty());

  EXPECT_FLOAT_EQ(var.uncertainty(), limit);

  Float expected_ratio = this->variable_.belief() / this->variable_.disbelief();
  Float ratio = var.belief() / var.disbelief();
  EXPECT_FLOAT_EQ(ratio, expected_ratio);
}

TYPED_TEST(BinomialOpinionTest, StringConversion)
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
