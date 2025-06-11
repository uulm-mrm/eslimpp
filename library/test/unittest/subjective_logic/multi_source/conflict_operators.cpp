#include <iostream>
#include <algorithm>

#include "gtest/gtest.h"

#include "subjective_logic_lib/multi_source/conflict_operators.hpp"

namespace subjective_logic::multisource
{

TEST(MultiSourceNoBaseConflictTest, JosangExampleCumFuse)
{
  OpinionNoBase op_c1(0.1, 0.3);
  OpinionNoBase op_c2(0.4, 0.2);
  OpinionNoBase op_c3(0.7, 0.1);

  float expected_acc_conflict{ 0 };
  expected_acc_conflict += op_c1.degree_of_conflict(op_c2);
  expected_acc_conflict += op_c1.degree_of_conflict(op_c3);
  expected_acc_conflict += op_c2.degree_of_conflict(op_c3);
  float expected_avg_conflict = expected_acc_conflict / 3;

  float acc_conflict = Conflict::conflict(Conflict::ConflictType::ACCUMULATE, op_c1, op_c2, op_c3);
  EXPECT_FLOAT_EQ(expected_acc_conflict, acc_conflict);

  float avg_conflict = Conflict::conflict(Conflict::ConflictType::AVERAGE, op_c1, op_c2, op_c3);
  EXPECT_FLOAT_EQ(expected_avg_conflict, avg_conflict);
}

using TestTypes = ::testing::Types<OpinionNoBase<2, float>,
                                   OpinionNoBase<3, float>,
                                   OpinionNoBase<6, float>,
                                   OpinionNoBase<2, double>,
                                   OpinionNoBase<3, double>,
                                   OpinionNoBase<6, double>>;

template <typename OpinionT>
class MultiSourceNoBaseConflictTest : public ::testing::Test
{
  //    using FloatT = typename OpinionT::FLOAT_t;
  //    static constexpr std::size_t N = OpinionT::SIZE;
};

TYPED_TEST_SUITE(MultiSourceNoBaseConflictTest, TestTypes);

TYPED_TEST(MultiSourceNoBaseConflictTest, MaxConflictMaxDim)
{
  using FloatT = typename TypeParam::FLOAT_t;
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TypeParam> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    opinions[idx].belief_masses()[idx] = 1.;
  }

  FloatT expected_acc_conflict = N * (N - 1) / 2.;
  FloatT acc_conflict = Conflict::conflict(Conflict::ConflictType::ACCUMULATE, opinions);
  EXPECT_FLOAT_EQ(expected_acc_conflict, acc_conflict);

  FloatT expected_avg_conflict = 1.;
  FloatT avg_conflict = Conflict::conflict(Conflict::ConflictType::AVERAGE, opinions);
  EXPECT_FLOAT_EQ(expected_avg_conflict, avg_conflict);
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceConflictSharesMaxConflict)
{
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TypeParam> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    opinions[idx].belief_masses()[idx] = 1.;
  }

  auto [avg_conflict, conflict_shares] =
      Conflict::conflict_shares<Conflict::RelationType::CONFLICT>(Conflict::ConflictType::AVERAGE, opinions);

  EXPECT_FLOAT_EQ(avg_conflict, 1.0F);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    if constexpr (N == 2)
    {
      EXPECT_FLOAT_EQ(conflict_shares[idx], 1.0);
    }
    else
    {
      EXPECT_FLOAT_EQ(conflict_shares[idx], 0.0);
    }
  }
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceConflictSharesZeroConflict)
{
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TypeParam> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    opinions[idx].belief_masses()[0] = 1.;
  }

  auto [avg_conflict, conflict_shares] =
      Conflict::conflict_shares<Conflict::RelationType::CONFLICT>(Conflict::ConflictType::AVERAGE, opinions);

  EXPECT_FLOAT_EQ(avg_conflict, 0.0F);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    EXPECT_FLOAT_EQ(conflict_shares[idx], 0.0);
  }

  auto [acc_conflict, conflict_shares2] =
      Conflict::conflict_shares<Conflict::RelationType::CONFLICT>(Conflict::ConflictType::ACCUMULATE, opinions);

  EXPECT_FLOAT_EQ(acc_conflict, 0.0F);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    EXPECT_FLOAT_EQ(conflict_shares2[idx], 0.0);
  }
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceConflictSharesOutlier)
{
  std::vector<TypeParam> opinions(3);
  opinions[0].belief_masses()[0] = 1.0;
  opinions[1].belief_masses()[0] = 1.0;
  opinions[2].belief_masses()[1] = 1.0;

  auto [avg_conflict, conflict_shares] =
      Conflict::conflict_shares<Conflict::RelationType::CONFLICT>(Conflict::ConflictType::AVERAGE, opinions);

  EXPECT_FLOAT_EQ(conflict_shares[0], -0.5);
  EXPECT_FLOAT_EQ(conflict_shares[1], -0.5);
  EXPECT_FLOAT_EQ(conflict_shares[2], 1.0);
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceBeliefConflictZeroConflict)
{
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TypeParam> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    opinions[idx].belief_masses()[0] = 1.;
  }

  auto [cum_conflict, conflict_shares] =
      Conflict::conflict_shares<Conflict::RelationType::CONFLICT>(Conflict::ConflictType::BELIEF_CUMULATIVE, opinions);
  auto [bc_conflict, conflict_shares2] = Conflict::conflict_shares<Conflict::RelationType::CONFLICT>(
      Conflict::ConflictType::BELIEF_BELIEF_CONSTRAINT, opinions);
  auto [avg_conflict, conflict_shares3] =
      Conflict::conflict_shares<Conflict::RelationType::CONFLICT>(Conflict::ConflictType::BELIEF_AVERAGE, opinions);

  EXPECT_FLOAT_EQ(cum_conflict, 0.0F);
  EXPECT_FLOAT_EQ(bc_conflict, 0.0F);
  EXPECT_FLOAT_EQ(avg_conflict, 0.0F);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    EXPECT_FLOAT_EQ(conflict_shares[idx], 0.0);
    EXPECT_FLOAT_EQ(conflict_shares2[idx], 0.0);
    EXPECT_FLOAT_EQ(conflict_shares3[idx], 0.0);
  }
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceUncertaintyDifferentialsZero)
{
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TypeParam> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    opinions[idx].belief_masses()[idx] = 1.;
  }

  auto uncertainty_differentials = Conflict::uncertainty_differentials(opinions);

  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    EXPECT_FLOAT_EQ(uncertainty_differentials[idx], 0.);
  }
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceUncertaintyDifferentialsEqual)
{
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TypeParam> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    opinions[idx].belief_masses()[idx] = 0.5;
  }

  auto uncertainty_differentials = Conflict::uncertainty_differentials(opinions);

  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    EXPECT_FLOAT_EQ(uncertainty_differentials[idx], 1 / static_cast<double>(opinions.size()));
  }
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceUncertaintyDifferentialsUnequal)
{
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TypeParam> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    opinions[idx].belief_masses()[idx] = 0.5;
  }
  opinions[0].belief_masses()[0] = 1.0;

  auto uncertainty_differentials = Conflict::uncertainty_differentials(opinions);

  EXPECT_FLOAT_EQ(uncertainty_differentials[0], 0);
  for (std::size_t idx{ 1 }; idx < N; ++idx)
  {
    EXPECT_FLOAT_EQ(uncertainty_differentials[idx], 1 / static_cast<double>(opinions.size() - 1));
  }
}

TYPED_TEST(MultiSourceNoBaseConflictTest, MultiSourceUncertaintyDifferentialsTrusted)
{
  static constexpr std::size_t N = TypeParam::SIZE;

  std::vector<TrustedOpinion<TypeParam>> opinions(N);
  for (std::size_t idx{ 0 }; idx < N; ++idx)
  {
    // trusts are always 2D, so assign belief masses to the first entry only
    opinions[idx].trust().belief_masses()[0] = 0.5;
  }
  opinions[0].trust().belief_masses()[0] = 1.0;

  auto uncertainty_differentials = Conflict::uncertainty_differentials(opinions);

  EXPECT_FLOAT_EQ(uncertainty_differentials[0], 0);
  for (std::size_t idx{ 1 }; idx < N; ++idx)
  {
    EXPECT_FLOAT_EQ(uncertainty_differentials[idx], 1 / static_cast<double>(opinions.size() - 1));
  }
}

}  // namespace subjective_logic::multisource
