#include <iostream>
#include <algorithm>

#include "gtest/gtest.h"

#include "subjective_logic_lib/multi_source/fusion_operators.hpp"

namespace subjective_logic::multisource
{

TEST(MultiSourceNoBaseFusionTest, JosangExampleCumFuse)
{
  OpinionNoBase op_c1(0.1, 0.3);
  OpinionNoBase op_c2(0.4, 0.2);
  OpinionNoBase op_c3(0.7, 0.1);

  auto result = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, op_c1, op_c2, op_c3);
  EXPECT_FLOAT_EQ(result.belief(), 0.6511628);
  EXPECT_FLOAT_EQ(result.disbelief(), 0.20930232);

  result = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, op_c2, op_c1, op_c3);
  EXPECT_FLOAT_EQ(result.belief(), 0.6511628);
  EXPECT_FLOAT_EQ(result.disbelief(), 0.20930232);

  result = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, op_c3, op_c1, op_c2);
  EXPECT_FLOAT_EQ(result.belief(), 0.6511628);
  EXPECT_FLOAT_EQ(result.disbelief(), 0.20930232);
}

TEST(MultiSourceNoBaseFusionTest, JosangExampleAvgFuse)
{
  OpinionNoBase op_c1(0.1, 0.3);
  OpinionNoBase op_c2(0.4, 0.2);
  OpinionNoBase op_c3(0.7, 0.1);

  auto result = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, op_c1, op_c2, op_c3);
  EXPECT_FLOAT_EQ(result.belief(), 0.5090909);
  EXPECT_FLOAT_EQ(result.disbelief(), 0.16363636);

  result = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, op_c2, op_c1, op_c3);
  EXPECT_FLOAT_EQ(result.belief(), 0.5090909);
  EXPECT_FLOAT_EQ(result.disbelief(), 0.16363636);

  result = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, op_c3, op_c1, op_c2);
  EXPECT_FLOAT_EQ(result.belief(), 0.5090909);
  EXPECT_FLOAT_EQ(result.disbelief(), 0.16363636);
}

using TestTypes = ::testing::Types<OpinionNoBase<2, float>,
                                   OpinionNoBase<3, float>,
                                   OpinionNoBase<6, float>,
                                   OpinionNoBase<2, double>,
                                   OpinionNoBase<3, double>,
                                   OpinionNoBase<6, double> >;

template <typename OpinionT>
class MultiSourceNoBaseFusionTest : public ::testing::Test
{
  //  using FloatT = typename OpinionT::FLOAT_t;
  //  static constexpr std::size_t N = OpinionT::SIZE;
};

TYPED_TEST_SUITE(MultiSourceNoBaseFusionTest, TestTypes);

TYPED_TEST(MultiSourceNoBaseFusionTest, CumFuseTwoVariablesVacuous)
{
  TypeParam var1{};
  TypeParam var2{};

  auto result_vacuous = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, var1, var2);
  auto expected_result = var1.cum_fuse(var2);

  EXPECT_FLOAT_EQ(result_vacuous.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result_vacuous.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, CumFuseTwoVariablesDogmatic)
{
  TypeParam var1{};
  TypeParam var2{};

  var1.belief_masses().front() = 1;
  var2.belief_masses().back() = 1;
  auto result_dogmatic = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, var1, var2);
  auto expected_result = var1.cum_fuse(var2);

  EXPECT_FLOAT_EQ(result_dogmatic.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result_dogmatic.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, CumFuseTwoVariables)
{
  TypeParam var1{};
  TypeParam var2{};

  var1.belief_masses().front() = 0.2;
  var2.belief_masses().back() = 0.5;
  auto result = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, var1, var2);
  auto expected_result = var1.cum_fuse(var2);

  EXPECT_FLOAT_EQ(result.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, CumFuseVariablesVacuous)
{
  TypeParam var1{};
  TypeParam var2{};
  TypeParam var3{};
  TypeParam var4{};

  auto result = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, var1, var2, var3, var4);
  auto expected_result = var1;

  EXPECT_FLOAT_EQ(result.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, CumFuseVariablesDogmatic)
{
  TypeParam var1{};
  var1.belief_masses().front() = 1.;
  TypeParam var2{};
  var2.belief_masses().front() = 1.;
  TypeParam var3{};
  var3.belief_masses().back() = 1.;
  TypeParam var4{};
  var4.belief_masses().back() = 1.;

  auto result = Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, var1, var2, var3, var4);
  TypeParam expected_result{};
  expected_result.belief_masses().front() = 0.5;
  expected_result.belief_masses().back() = 0.5;

  EXPECT_FLOAT_EQ(result.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, CumFuseVariables)
{
  TypeParam var1{};
  var1.belief_masses().front() = .2;
  TypeParam var2{};
  var2.belief_masses().front() = .5;
  TypeParam var3{};
  var3.belief_masses().back() = .1;
  TypeParam var4{};
  var4.belief_masses().back() = .3;

  std::vector<TypeParam> opinions = { var1, var2, var3, var4 };
  std::vector<TypeParam> results;

  auto comperator = [](TypeParam tp1, TypeParam tp2) { return tp1.belief_masses()[0] < tp2.belief_masses()[0]; };
  do
  {
    results.push_back(Fusion::fuse_opinions(Fusion::FusionType::CUMULATIVE, opinions));
  } while (std::ranges::next_permutation(opinions, comperator).found);

  for (std::size_t idx{ 1 }; idx < results.size(); ++idx)
  {
    EXPECT_FLOAT_EQ(results[0].uncertainty(), results[idx].uncertainty());
    for (std::size_t mass_idx{ 0 }; mass_idx < TypeParam::SIZE; ++mass_idx)
    {
      EXPECT_FLOAT_EQ(results[0].belief_masses()[mass_idx], results[idx].belief_masses()[mass_idx]);
    }
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, AvgFuseTwoVariablesVacuous)
{
  TypeParam var1{};
  TypeParam var2{};

  auto result_vacuous = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, var1, var2);
  auto expected_result = var1.average_fuse(var2);

  EXPECT_FLOAT_EQ(result_vacuous.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result_vacuous.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, AvgFuseTwoVariablesDogmatic)
{
  TypeParam var1{};
  TypeParam var2{};

  var1.belief_masses().front() = 1;
  var2.belief_masses().back() = 1;
  auto result_dogmatic = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, var1, var2);
  auto expected_result = var1.average_fuse(var2);

  EXPECT_FLOAT_EQ(result_dogmatic.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result_dogmatic.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, AvgFuseTwoVariables)
{
  TypeParam var1{};
  TypeParam var2{};

  var1.belief_masses().front() = 0.2;
  var2.belief_masses().back() = 0.5;
  auto result = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, var1, var2);
  auto expected_result = var1.average_fuse(var2);

  EXPECT_FLOAT_EQ(result.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, AvgFuseVariablesVacuous)
{
  TypeParam var1{};
  TypeParam var2{};
  TypeParam var3{};
  TypeParam var4{};

  auto result = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, var1, var2, var3, var4);
  auto expected_result = var1;

  EXPECT_FLOAT_EQ(result.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, AvgFuseVariablesDogmatic)
{
  TypeParam var1{};
  var1.belief_masses().front() = 1.;
  TypeParam var2{};
  var2.belief_masses().front() = 1.;
  TypeParam var3{};
  var3.belief_masses().back() = 1.;
  TypeParam var4{};
  var4.belief_masses().back() = 1.;

  auto result = Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, var1, var2, var3, var4);
  TypeParam expected_result{};
  expected_result.belief_masses().front() = 0.5;
  expected_result.belief_masses().back() = 0.5;

  EXPECT_FLOAT_EQ(result.uncertainty(), expected_result.uncertainty());
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(expected_result.belief_masses()[idx], result.belief_masses()[idx]);
  }
}

TYPED_TEST(MultiSourceNoBaseFusionTest, AvgFuseVariables)
{
  TypeParam var1{};
  var1.belief_masses().front() = .2;
  TypeParam var2{};
  var2.belief_masses().front() = .5;
  TypeParam var3{};
  var3.belief_masses().back() = .1;
  TypeParam var4{};
  var4.belief_masses().back() = .3;

  std::vector<TypeParam> opinions = { var1, var2, var3, var4 };
  std::vector<TypeParam> results;

  auto comperator = [](TypeParam tp1, TypeParam tp2) { return tp1.belief_masses()[0] < tp2.belief_masses()[0]; };
  do
  {
    results.push_back(Fusion::fuse_opinions(Fusion::FusionType::AVERAGE, opinions));
  } while (std::ranges::next_permutation(opinions, comperator).found);

  for (std::size_t idx{ 1 }; idx < results.size(); ++idx)
  {
    EXPECT_FLOAT_EQ(results[0].uncertainty(), results[idx].uncertainty());
    for (std::size_t mass_idx{ 0 }; mass_idx < TypeParam::SIZE; ++mass_idx)
    {
      EXPECT_FLOAT_EQ(results[0].belief_masses()[mass_idx], results[idx].belief_masses()[mass_idx]);
    }
  }
}

}  // namespace subjective_logic::multisource
