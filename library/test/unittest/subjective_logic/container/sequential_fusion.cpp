#include "gtest/gtest.h"

#include "subjective_logic_lib/container/long_short_term_memory.hpp"
#include "subjective_logic_lib/types/fusion_types.hpp"

namespace subjective_logic::container
{
using TestTypes =
    ::testing::Types<OpinionNoBase<3, float>, OpinionNoBase<6, double>, Opinion<3, double>, Opinion<6, float> >;

template <typename OpinionT>
class SequentialFusionTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
  }
};

TYPED_TEST_SUITE(SequentialFusionTest, TestTypes);

TYPED_TEST(SequentialFusionTest, TwoOpsABF)
{
  using FloatT = typename TypeParam::FLOAT_t;
  using SF = multisource::SequentialFusion;

  constexpr FloatT weight{ 1.0 };

  TypeParam a{};
  a.belief_masses()[0] = 0.9;

  TypeParam b{};
  b.belief_masses()[1] = 0.5;
  auto [fusion_result, updated_weight] = SF::average_fusion_operator(a, b, weight, 1.0);

  auto expected = a.average_fuse(b);
  EXPECT_EQ(fusion_result, expected);
  EXPECT_FLOAT_EQ(updated_weight, weight + 1.0);

  // One of the opinions is dogmatic
  TypeParam dogmatic{};
  dogmatic.belief_masses()[0] = 1.0;
  std::tie(fusion_result, updated_weight) = SF::average_fusion_operator(dogmatic, b, weight, 1.0);
  EXPECT_EQ(fusion_result, dogmatic);
  EXPECT_FLOAT_EQ(updated_weight, weight + 1.0);

  std::tie(fusion_result, updated_weight) = SF::average_fusion_operator(a, dogmatic, weight, 1.0);
  EXPECT_EQ(fusion_result, dogmatic);
  EXPECT_FLOAT_EQ(updated_weight, weight + 1.0);

  // Both opinions are dogmatic
  TypeParam dogmatic_inv{};
  dogmatic_inv.belief_masses()[1] = 1.0;
  std::tie(fusion_result, updated_weight) = SF::average_fusion_operator(dogmatic, dogmatic_inv, weight, 1.0);
  EXPECT_FLOAT_EQ(fusion_result.belief_masses()[0], 0.5);
  EXPECT_FLOAT_EQ(fusion_result.belief_masses()[1], 0.5);

  // test that weight converges as expected
  constexpr FloatT discount{ 0.8 };
  updated_weight = 1.0;
  for (std::size_t idx = 0; idx < 1000; ++idx)
  {
    std::tie(fusion_result, updated_weight) = SF::average_fusion_operator(a, b, updated_weight, discount);
  }
  EXPECT_NEAR(updated_weight, 1 / (1 - discount), 1e-5);

  // TODO(@deuscher) - test case using discount
}

TYPED_TEST(SequentialFusionTest, TwoOpsCBF)
{
  using FloatT = typename TypeParam::FLOAT_t;
  using SF = multisource::SequentialFusion;

  constexpr FloatT weight{ 1.0 };

  TypeParam a{};
  a.belief_masses()[0] = 0.9;

  TypeParam b{};
  b.belief_masses()[1] = 0.5;
  auto fusion_result = SF::cumulative_fusion_operator(a, b, weight, 1.0);
  auto expected = a.cum_fuse(b);
  EXPECT_EQ(fusion_result, expected);

  // One of the opinions is dogmatic
  TypeParam dogmatic{};
  dogmatic.belief_masses()[0] = 1.0;
  fusion_result = SF::cumulative_fusion_operator(dogmatic, b, weight, 1.0);
  EXPECT_EQ(fusion_result, dogmatic);

  fusion_result = SF::cumulative_fusion_operator(a, dogmatic, weight, 1.0);
  EXPECT_EQ(fusion_result, dogmatic);

  // Both opinions are dogmatic
  TypeParam dogmatic_inv{};
  dogmatic_inv.belief_masses()[1] = 1.0;
  fusion_result = SF::cumulative_fusion_operator(dogmatic, dogmatic_inv, weight, 1.0);
  EXPECT_FLOAT_EQ(fusion_result.belief_masses()[0], 0.5);
  EXPECT_FLOAT_EQ(fusion_result.belief_masses()[1], 0.5);
}

TYPED_TEST(SequentialFusionTest, TwoOpsWBF)
{
  using FloatT = typename TypeParam::FLOAT_t;
  using SF = multisource::SequentialFusion;

  constexpr FloatT weight{ 1.0 };
  TypeParam a{};
  a.belief_masses()[0] = 0.9;

  TypeParam b{};
  b.belief_masses()[1] = 0.5;
  auto [fusion_result, updated_weight] = SF::weighted_fusion_operator(a, b, (1 - a.uncertainty()), 1.0);

  auto expected = a.wb_fuse(b);
  EXPECT_EQ(fusion_result, expected);
  EXPECT_FLOAT_EQ(updated_weight, (1 - a.uncertainty()) + (1 - b.uncertainty()));

  // One of the opinions is dogmatic
  TypeParam dogmatic{};
  dogmatic.belief_masses()[0] = 1.0;
  std::tie(fusion_result, updated_weight) = SF::weighted_fusion_operator(dogmatic, b, weight, 1.0);
  EXPECT_EQ(fusion_result, dogmatic);
  EXPECT_FLOAT_EQ(updated_weight, weight + (1 - b.uncertainty()));

  std::tie(fusion_result, updated_weight) = SF::weighted_fusion_operator(a, dogmatic, weight, 1.0);
  EXPECT_EQ(fusion_result, dogmatic);
  EXPECT_FLOAT_EQ(updated_weight, weight + (1 - dogmatic.uncertainty()));

  // Both opinions are dogmatic
  TypeParam dogmatic_inv{};
  dogmatic_inv.belief_masses()[1] = 1.0;
  std::tie(fusion_result, updated_weight) = SF::weighted_fusion_operator(dogmatic, dogmatic_inv, weight, 1.0);
  EXPECT_FLOAT_EQ(fusion_result.belief_masses()[0], 0.5);
  EXPECT_FLOAT_EQ(fusion_result.belief_masses()[1], 0.5);
}

}  // namespace subjective_logic::container