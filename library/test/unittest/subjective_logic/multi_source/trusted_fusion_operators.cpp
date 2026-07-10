#include <iostream>
#include <algorithm>

#include "gtest/gtest.h"

#include "subjective_logic_lib/multi_source/trusted_fusion_operators.hpp"

namespace subjective_logic::multisource
{

TEST(MultiSourceTrustedFusionTest, JosangExampleCumFuse)
{
  Trust<double> a_c1{ 0.3, 0.0, 0.9 };
  Trust<double> a_c2{ 0.7, 0.0, 0.9 };
  Trust<double> a_c3{ 0.4, 0.1, 0.9 };

  Opinion c1_x{ 1.0, 0.0, 0.1 };
  Opinion c2_x{ 0.0, 1.0, 0.1 };
  Opinion c3_x{ 1.0, 0.0, 0.1 };

  TrustedOpinion<Opinion<2, double>> a_c1_x{ a_c1, c1_x };
  TrustedOpinion<Opinion<2, double>> a_c2_x{ a_c2, c2_x };
  TrustedOpinion<Opinion<2, double>> a_c3_x{ a_c3, c3_x };

  std::vector<TrustedOpinion<Opinion<2, double>>> t_ops{ a_c1_x, a_c2_x, a_c3_x };

  auto cum_fused_no_revision = TrustedFusion::fuse_opinions(FusionType::CUMULATIVE, t_ops);

  // since the example is copied from a paper table, numbers are rather a rough estimate
  EXPECT_NEAR(cum_fused_no_revision.belief(), 0.36, 0.05);
  EXPECT_NEAR(cum_fused_no_revision.disbelief(), 0.62, 0.05);
  EXPECT_NEAR(cum_fused_no_revision.uncertainty(), 0.02, 0.05);

  auto avg_fused_no_revision = TrustedFusion::fuse_opinions(FusionType::AVERAGE, t_ops);

  // since the example is copied from a paper table, numbers are rather a rough estimate
  EXPECT_NEAR(avg_fused_no_revision.belief(), 0.35, 0.05);
  EXPECT_NEAR(avg_fused_no_revision.disbelief(), 0.60, 0.05);
  EXPECT_NEAR(avg_fused_no_revision.uncertainty(), 0.05, 0.05);

  // for the trust revision josang always uses the average reference fusion
  auto cum_fused_cum_revision = TrustedFusion::fuse_opinions(FusionType::CUMULATIVE,
                                                             RelationType::CONFLICT,
                                                             TrustRevisionType::REFERENCE_FUSION,
                                                             ConflictType::BELIEF_AVERAGE,
                                                             t_ops);
#ifdef BELIEF_REVISION_FOLLOWING_JOSAN
  // since the example is copied from a paper table, numbers are rather a rough estimate
  EXPECT_NEAR(cum_fused_cum_revision.belief(), 0.06, 0.06);
  EXPECT_NEAR(cum_fused_cum_revision.disbelief(), 0.91, 0.06);
  EXPECT_NEAR(cum_fused_cum_revision.uncertainty(), 0.03, 0.06);
#endif

  auto avg_fused_avg_revision = TrustedFusion::fuse_opinions(FusionType::AVERAGE,
                                                             RelationType::CONFLICT,
                                                             TrustRevisionType::REFERENCE_FUSION,
                                                             ConflictType::BELIEF_AVERAGE,
                                                             t_ops);

#ifdef BELIEF_REVISION_FOLLOWING_JOSAN
  // since the example is copied from a paper table, numbers are rather a rough estimate
  EXPECT_NEAR(avg_fused_avg_revision.belief(), 0.06, 0.05);
  EXPECT_NEAR(avg_fused_avg_revision.disbelief(), 0.86, 0.05);
  EXPECT_NEAR(avg_fused_avg_revision.uncertainty(), 0.08, 0.05);
#endif
}

using TestTypes = ::testing::Types<TrustedOpinion<OpinionNoBase<2, float>>>;

template <typename OpinionT>
class MultiSourceTrustedFusionTest : public ::testing::Test
{
  //  using FloatT = typename OpinionT::FLOAT_t;
  //  static constexpr std::size_t N = OpinionT::SIZE;
};

TYPED_TEST_SUITE(MultiSourceTrustedFusionTest, TestTypes);

TYPED_TEST(MultiSourceTrustedFusionTest, CumFuseTwoVariablesVacuous)
{
  using OpinionT = typename TypeParam::OpinionT;
  TypeParam var1{};
  TypeParam var2{};

  EXPECT_FLOAT_EQ(var1.opinion().uncertainty(), 1.0);

  std::vector<TypeParam> vec{ var1, var2 };
  OpinionT cum_fused = TrustedFusion::fuse_opinions(
      FusionType::CUMULATIVE, RelationType::CONFLICT, TrustRevisionType::SHARES, ConflictType::AVERAGE, vec);

  EXPECT_FLOAT_EQ(cum_fused.uncertainty(), 1.0);
}

TYPED_TEST(MultiSourceTrustedFusionTest, CumFuseTwoVariables)
{
  using OpinionT = typename TypeParam::OpinionT;
  using TrustT = typename TypeParam::TrustT;
  OpinionT opin1{};
  opin1.belief_mass(0) = 0.4;
  opin1.belief_mass(1) = 0.2;
  TrustT trust1{ 0.5, 0.5 };
  OpinionT opin2{};
  opin1.belief_mass(0) = 0.3;
  opin1.belief_mass(1) = 0.5;
  TrustT trust2{ 0.5, 0.5 };
  TypeParam var1{ trust1, opin1 };
  TypeParam var2{ trust2, opin2 };

  std::vector<TypeParam> vec{ var1, var2 };
  OpinionT cum_fused = TrustedFusion::fuse_opinions(
      FusionType::CUMULATIVE, RelationType::CONFLICT, TrustRevisionType::SHARES, ConflictType::AVERAGE, vec);
  // values are within a range, thus, not nan
  EXPECT_GE(cum_fused.belief_mass(0), 0.01);
  EXPECT_LE(cum_fused.belief_mass(0), 0.99);
  EXPECT_GE(cum_fused.belief_mass(1), 0.01);
  EXPECT_LE(cum_fused.belief_mass(1), 0.99);
}

TYPED_TEST(MultiSourceTrustedFusionTest, CumFuseTwoVariablesCudaImpl)
{
  using OpinionT = typename TypeParam::OpinionT;
  using TrustT = typename TypeParam::TrustT;
  using BeliefType = typename TrustT::BeliefType;
  OpinionT opin1{};
  opin1.belief_mass(0) = 0.4;
  opin1.belief_mass(1) = 0.0;
  TrustT trust1{ BeliefType{ 0.0, 0.0 }, BeliefType{ 1.0, 0.0 } };
  OpinionT opin2{};
  opin2.belief_mass(0) = 0.0;
  opin2.belief_mass(1) = 0.5;
  TrustT trust2{ BeliefType{ 0.0, 0.0 }, BeliefType{ 1.0, 0.0 } };
  TypeParam var1{ trust1, opin1 };
  TypeParam var2{ trust2, opin2 };

  Array<2, TypeParam> vec{ var1, var2 };
  auto discounted_opins = TypeParam::extractDiscountedOpinions(vec);
  OpinionT ref_fusion_no_tr = Fusion::fuse_opinions(FusionType::CUMULATIVE, discounted_opins);

  using WeightTypes = multisource::TrustedFusion::WeightedTypes;
  Array<1, WeightTypes> weights{ WeightTypes{ subjective_logic::RelationType::CONFLICT,
                                              subjective_logic::TrustRevisionType::SHARES,
                                              subjective_logic::ConflictType::AVERAGE,
                                              0.0 } };

  OpinionT cum_fused = TrustedFusion::fuse_opinions(FusionType::CUMULATIVE, weights, vec);

  EXPECT_EQ(cum_fused, ref_fusion_no_tr);

  weights[0].weight = 1.0;
  OpinionT cum_fused_with_tr = TrustedFusion::fuse_opinions(FusionType::CUMULATIVE, weights, vec);

  EXPECT_NE(cum_fused_with_tr, ref_fusion_no_tr);
}

}  // namespace subjective_logic::multisource