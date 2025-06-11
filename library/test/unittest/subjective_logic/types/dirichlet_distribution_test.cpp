#include "gtest/gtest.h"

#include "subjective_logic_lib/types/dirichlet_distribution.hpp"

namespace subjective_logic
{

TEST(DirichletDistributionTest, ValuesCtor)
{
  DirichletDistribution<2, float>{ 1, 2 };
  DirichletDistribution<2, float>(1.F, 2.F);
  DirichletDistribution<3, double>(5., 2., 4.);
  DirichletDistribution<10, double>(5., 2., 1., 2., 3., 4., 5., 6., 7., 8.);
}

// values for this test have been generated using scipy.stats
TEST(DirichletDistributionTest, SampleEvaluation)
{
  Array<3, float> quantiles3{ 0.2, 0.2, 0.6 };
  Array<3, float> alphas3{ 2, 4, 8 };
  float expected_value = 9.223107379199973;
  DirichletDistribution<3> dist3{ alphas3 };
  EXPECT_NEAR(dist3.evaluate(quantiles3), expected_value, EPS_v<float>);

  alphas3 = Array<3, float>{ 8., 4., 8. };
  expected_value = 2.2879209089138697;
  dist3 = DirichletDistribution<3>{ alphas3 };
  EXPECT_NEAR(dist3.evaluate(quantiles3), expected_value, EPS_v<float>);

  Array<2, float> quantiles2{ 0.3, 0.7 };
  Array<2, float> alphas2{ 0.00001, 8 };
  expected_value = 2.7451814602564156e-06;
  DirichletDistribution<2> dist2{ alphas2 };
  EXPECT_NEAR(dist2.evaluate(quantiles2), expected_value, EPS_v<float>);

  alphas2 = Array<2, float>{ 4, 8 };
  expected_value = 2.9351072519999994;
  dist2 = DirichletDistribution<2>{ alphas2 };
  EXPECT_NEAR(dist2.evaluate(quantiles2), expected_value, EPS_v<float>);
}

// values for this test have been generated using scipy.stats
TEST(DirichletDistributionTest, SampleMean)
{
  Array<2, float> alphas2{ 1, 8 };
  Array<2, float> expected2{ 0.11111111, 0.88888889 };
  DirichletDistribution<2> dist2{ alphas2 };
  auto mean2 = dist2.mean();
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_NEAR(mean2[idx], expected2[idx], EPS_v<float>);
  }
  EXPECT_FLOAT_EQ(mean2[0], dist2.mean_binomial());

  Array<4, float> alphas4{ 1, 8, 2, 5 };
  Array<4, float> expected4{ 0.0625, 0.5, 0.125, 0.3125 };
  DirichletDistribution<4> dist4{ alphas4 };
  auto mean4 = dist4.mean();
  for (std::size_t idx{ 0 }; idx < 4; ++idx)
  {
    EXPECT_NEAR(mean4[idx], expected4[idx], EPS_v<float>);
  }
}

// values for this test have been generated using scipy.stats
TEST(DirichletDistributionTest, SampleVariance)
{
  Array<2, float> alphas2{ 1, 8 };
  Array<2, float> expected2{ 0.00987654, 0.00987654 };
  DirichletDistribution<2> dist2{ alphas2 };
  auto variance2 = dist2.variance();
  for (std::size_t idx{ 0 }; idx < 2; ++idx)
  {
    EXPECT_NEAR(variance2[idx], expected2[idx], EPS_v<float>);
  }

  Array<4, float> alphas4{ 1, 8, 2, 5 };
  Array<4, float> expected4{ 0.00344669, 0.01470588, 0.00643382, 0.01263787 };
  DirichletDistribution<4> dist4{ alphas4 };
  auto variance4 = dist4.variance();
  for (std::size_t idx{ 0 }; idx < 4; ++idx)
  {
    EXPECT_NEAR(variance4[idx], expected4[idx], EPS_v<float>);
  }
}

using TestTypes = ::testing::Types<DirichletDistribution<3, float>,
                                   DirichletDistribution<6, float>,
                                   DirichletDistribution<3, double>,
                                   DirichletDistribution<6, double>>;

template <typename DirichletT>
class DirichletDistributionTest : public ::testing::Test
{
  // using FloatT = typename DirichletT::FLOAT_t;
  // static constexpr std::size_t N = DirichletT::SIZE;

protected:
  void SetUp() override
  {
  }

  DirichletT distribution_;
};

TYPED_TEST_SUITE(DirichletDistributionTest, TestTypes);

TYPED_TEST(DirichletDistributionTest, Ctor)
{
  using FloatT = typename TypeParam::FLOAT_t;
  TypeParam var{};
  auto alphas = var.alphas();
  FloatT equal_distr{ static_cast<FloatT>(1.0) / static_cast<FloatT>(TypeParam::SIZE) };
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(var.evidences()[idx], 0.);
    EXPECT_FLOAT_EQ(var.priors()[idx], equal_distr);
    EXPECT_FLOAT_EQ(alphas[idx], TypeParam::SIZE * equal_distr);
  }

  using WeightType = typename TypeParam::WeightType;
  WeightType weights;
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    weights[idx] = idx + 1;
  }
  TypeParam dist{ weights };

  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(dist.evidences()[idx], idx + 1 - TypeParam::SIZE * equal_distr);
    EXPECT_FLOAT_EQ(dist.priors()[idx], equal_distr);
    EXPECT_FLOAT_EQ(dist.alphas()[idx], idx + 1);
  }

  TypeParam dist2{ weights, WeightType{ equal_distr } };
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(dist2.evidences()[idx], idx + 1);
    EXPECT_FLOAT_EQ(dist2.priors()[idx], equal_distr);
    EXPECT_FLOAT_EQ(dist2.alphas()[idx], idx + 1 + TypeParam::SIZE * equal_distr);
  }
}

TYPED_TEST(DirichletDistributionTest, Accessors)
{
  using FloatT = typename TypeParam::FLOAT_t;
  FloatT equal_distr{ static_cast<FloatT>(1.0) / static_cast<FloatT>(TypeParam::SIZE) };

  const TypeParam const_var;
  const auto& const_alphas = const_var.alphas();
  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(this->distribution_.evidences()[idx], 0.);
    EXPECT_FLOAT_EQ(const_var.evidences()[idx], 0.);
    EXPECT_FLOAT_EQ(this->distribution_.priors()[idx], equal_distr);
    EXPECT_FLOAT_EQ(const_var.priors()[idx], equal_distr);
    EXPECT_FLOAT_EQ(this->distribution_.alphas()[idx], TypeParam::SIZE * equal_distr);
    EXPECT_FLOAT_EQ(const_alphas[idx], TypeParam::SIZE * equal_distr);
  }

  this->distribution_.evidences()[0] += 1;
  EXPECT_FLOAT_EQ(this->distribution_.evidences()[0], 1);
}

TYPED_TEST(DirichletDistributionTest, Conversion)
{
  using FloatT = typename TypeParam::FLOAT_t;
  constexpr std::size_t size = TypeParam::SIZE;
  using WeightType = typename TypeParam::WeightType;
  WeightType alphas{ 2 };

  TypeParam distr{ alphas };
  auto opinion = static_cast<Opinion<size, FloatT>>(distr);
  auto opinion_no_base = static_cast<OpinionNoBase<size, FloatT>>(distr);
  auto convert_back = static_cast<TypeParam>(opinion);
  auto convert_back_no_base = static_cast<TypeParam>(opinion_no_base);

  for (std::size_t idx{ 0 }; idx < TypeParam::SIZE; ++idx)
  {
    EXPECT_FLOAT_EQ(distr.alphas()[idx], convert_back.alphas()[idx]);
    EXPECT_FLOAT_EQ(distr.alphas()[idx], convert_back_no_base.alphas()[idx]);
  }
}

}  // namespace subjective_logic