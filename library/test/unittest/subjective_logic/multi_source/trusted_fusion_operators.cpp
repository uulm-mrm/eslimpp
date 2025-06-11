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

  auto cum_fused_no_revision = TrustedFusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, t_ops);

  // since the example is copied from a paper table, numbers are rather a rough estimate
  EXPECT_NEAR(cum_fused_no_revision.belief(), 0.36, 0.05);
  EXPECT_NEAR(cum_fused_no_revision.disbelief(), 0.62, 0.05);
  EXPECT_NEAR(cum_fused_no_revision.uncertainty(), 0.02, 0.05);

  auto avg_fused_no_revision = TrustedFusion::fuse_opinions(Fusion::FusionType::AVERAGE, t_ops);

  // since the example is copied from a paper table, numbers are rather a rough estimate
  EXPECT_NEAR(avg_fused_no_revision.belief(), 0.35, 0.05);
  EXPECT_NEAR(avg_fused_no_revision.disbelief(), 0.60, 0.05);
  EXPECT_NEAR(avg_fused_no_revision.uncertainty(), 0.05, 0.05);

  // for the trust revision josang always uses the average reference fusion
  auto cum_fused_cum_revision = TrustedFusion::fuse_opinions(Fusion::FusionType::CUMULATIVE,
                                                             TrustRevision::TrustRevisionType::REFERENCE_FUSION,
                                                             Conflict::ConflictType::BELIEF_AVERAGE,
                                                             t_ops);
#ifdef BELIEF_REVISION_FOLLOWING_JOSAN
  // since the example is copied from a paper table, numbers are rather a rough estimate
  EXPECT_NEAR(cum_fused_cum_revision.belief(), 0.06, 0.06);
  EXPECT_NEAR(cum_fused_cum_revision.disbelief(), 0.91, 0.06);
  EXPECT_NEAR(cum_fused_cum_revision.uncertainty(), 0.03, 0.06);
#endif

  auto avg_fused_avg_revision = TrustedFusion::fuse_opinions(Fusion::FusionType::AVERAGE,
                                                             TrustRevision::TrustRevisionType::REFERENCE_FUSION,
                                                             Conflict::ConflictType::BELIEF_AVERAGE,
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
  OpinionT cum_fused = TrustedFusion::fuse_opinions(Fusion::FusionType::CUMULATIVE,
                                                    TrustRevision::TrustRevisionType::CONFLICT_SHARES,
                                                    Conflict::ConflictType::AVERAGE,
                                                    vec);

  //  std::vector<TrustedFusion::WeightedTypes> weighted_types;
  //  weighted_types.push_back({TrustRevision::TrustRevisionType::CONFLICT_SHARES, Conflict::ConflictType::AVERAGE,
  //  .5}); weighted_types.push_back({TrustRevision::TrustRevisionType::NORMAL, Conflict::ConflictType::ACCUMULATE,
  //  .5});
  //
  //  OpinionT cum_fused2 = TrustedFusion::fuse(
  //      Fusion::FusionType::CUMULATIVE,
  //      weighted_types,
  //      vec
  //  );
}

}  // namespace subjective_logic::multisource