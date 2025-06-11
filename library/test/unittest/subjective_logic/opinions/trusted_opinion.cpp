#include "gtest/gtest.h"

#include "subjective_logic_lib/opinions/trusted_opinion.hpp"

namespace subjective_logic
{
using TestTypes =
    ::testing::Types<OpinionNoBase<2, float>, OpinionNoBase<4, double>, Opinion<2, double>, Opinion<4, float>>;

template <typename OpinionT>
class TrustedOpinionTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
  }
};

TYPED_TEST_SUITE(TrustedOpinionTest, TestTypes);

TYPED_TEST(TrustedOpinionTest, ExtracOpinionsAndTrust)
{
  using FloatT = TypeParam::FLOAT_t;
  std::size_t N = TypeParam::SIZE;

  Trust<FloatT> default_trust{ 1.0, 0. };
  TypeParam default_opinion{};

  constexpr std::size_t num_ops{ 10 };
  std::vector<TrustedOpinion<TypeParam>> trusted_ops;
  trusted_ops.reserve(num_ops);
  for (std::size_t idx{ 0 }; idx < num_ops; ++idx)
  {
    auto ops = default_opinion;
    ops.belief_masses()[idx % N] = 0.5;
    auto trust = default_trust;
    trust.belief() = 1.0 / N * idx;
    trusted_ops.emplace_back(default_trust, default_opinion);
  }

  const auto& const_trusted_ops = trusted_ops;

  auto const_opinions = TrustedOpinion<TypeParam>::extractOpinions(const_trusted_ops);
  auto opinions = TrustedOpinion<TypeParam>::extractOpinionsRef(trusted_ops);

  auto const_trusts = TrustedOpinion<TypeParam>::extractTrusts(const_trusted_ops);
  auto trusts = TrustedOpinion<TypeParam>::extractTrustsRef(trusted_ops);

  for (std::size_t idx{ 0 }; idx < num_ops; ++idx)
  {
    auto& expected_t_op = trusted_ops[idx];
    auto& expected_opinion = expected_t_op.opinion();
    auto& expected_trust = expected_t_op.trust();

    auto const_opinion = const_opinions[idx];
    auto& opinion = opinions[idx];

    auto const_trust = const_trusts[idx];
    auto& trust = trusts[idx];

    EXPECT_EQ(const_opinion, expected_opinion);
    EXPECT_EQ(opinion, expected_opinion);

    EXPECT_EQ(const_trust, expected_trust);
    EXPECT_EQ(trust, expected_trust);

    opinion.get().belief_masses() = TypeParam::VacuousBeliefDistr();
    EXPECT_EQ(opinion, expected_opinion);

    trust.get() = Trust<FloatT>::VacuousTrust();
    EXPECT_EQ(trust, expected_trust);
  }
}

TYPED_TEST(TrustedOpinionTest, Ctor)
{
  using FloatT = TypeParam::FLOAT_t;

  Trust<FloatT> expected_trust{ 1.0, 0. };
  TypeParam expected_opinion;
  expected_opinion.belief_masses()[0] = 1.;

  TrustedOpinion<TypeParam> to1{};
  EXPECT_EQ(to1.trust(), Trust<FloatT>{});
  EXPECT_EQ(to1.opinion(), TypeParam{});

  TrustedOpinion to2{ expected_trust, expected_opinion };
  EXPECT_EQ(expected_trust, to2.trust());
  EXPECT_EQ(expected_opinion, to2.opinion());
}

TYPED_TEST(TrustedOpinionTest, IsValid)
{
  using FloatT = TypeParam::FLOAT_t;
  Trust<FloatT> trust;
  TypeParam opinion;

  TrustedOpinion to1{ trust, opinion };
  EXPECT_TRUE(to1.is_valid());

  auto to2 = to1;
  to2.opinion().belief_masses()[0] = 2.;
  EXPECT_FALSE(to2.is_valid());

  auto to3 = to1;
  to3.trust().belief_masses()[0] = 2.;
  EXPECT_FALSE(to3.is_valid());
}

TYPED_TEST(TrustedOpinionTest, DiscountedOpinion)
{
  using FloatT = TypeParam::FLOAT_t;
  using BeliefType = typename Trust<FloatT>::BeliefType;

  Trust<FloatT> trust1{ 1.0, 0. };
  TypeParam opinion;
  opinion.belief_masses()[0] = 1.;

  TrustedOpinion to1{ trust1, opinion };
  auto dis_opinion = to1.discounted_opinion();
  EXPECT_EQ(dis_opinion, opinion);

  FloatT discount = 0.5;
  Trust<FloatT> trust2{ BeliefType{ discount, 0. }, BeliefType{ 0., 0. } };
  TrustedOpinion to2{ trust2, opinion };
  auto dis_opinion2 = to2.discounted_opinion();
  EXPECT_EQ(dis_opinion2, opinion.trust_discount(discount));
}

TYPED_TEST(TrustedOpinionTest, TrustAccess)
{
  using FloatT = TypeParam::FLOAT_t;
  //  using BeliefType = typename Trust<FloatT>::BeliefType;

  Trust<FloatT> trust{ 0.8, 0.2 };
  TypeParam opinion{};

  TrustedOpinion<TypeParam> trusted_opinion(trust, opinion);

  auto& trust_access = trusted_opinion.trust();
  auto const& const_trust_access = static_cast<const TrustedOpinion<TypeParam>&>(trusted_opinion).trust();

  EXPECT_EQ(trust_access, trust);
  EXPECT_EQ(const_trust_access, trust);

  trust_access.belief() = 0.9;
  trust_access.disbelief() = 0.1;

  EXPECT_EQ(trust_access, const_trust_access);
}

TYPED_TEST(TrustedOpinionTest, OpinionAccess)
{
  using FloatT = TypeParam::FLOAT_t;

  Trust<FloatT> trust{};
  TypeParam opinion{};
  opinion.belief_masses()[0] = 1.0;

  TrustedOpinion<TypeParam> trusted_opinion(trust, opinion);

  auto& opinion_access = trusted_opinion.opinion();
  auto const& const_opinion_access = static_cast<const TrustedOpinion<TypeParam>&>(trusted_opinion).opinion();

  EXPECT_EQ(opinion_access, opinion);
  EXPECT_EQ(const_opinion_access, opinion);

  opinion.belief_masses()[0] = 0.;
  opinion.belief_masses()[1] = 1.;

  EXPECT_EQ(opinion_access, const_opinion_access);
}

TYPED_TEST(TrustedOpinionTest, TrustRevision)
{
  using FloatT = TypeParam::FLOAT_t;

  Trust<FloatT> trust_1{ 0.8, 0. };
  Trust<FloatT> trust_2{ 0.5, 0.1 };

  TypeParam opinion_1{};
  opinion_1.belief_masses()[0] = 1.0;
  TypeParam opinion_2{};
  opinion_2.belief_masses()[1] = 1.0;

  TrustedOpinion<TypeParam> t_op_1{ trust_1, opinion_1 };
  TrustedOpinion<TypeParam> t_op_2{ trust_2, opinion_2 };

  auto conflict = opinion_1.degree_of_conflict(opinion_2);
  auto uncert_1 = trust_1.uncertainty_differential(trust_2);
  auto uncert_2 = trust_2.uncertainty_differential(trust_1);

  auto revision_fac_1 = uncert_1 * conflict;
  auto revision_fac_2 = uncert_2 * conflict;

  auto expected_op_1 = trust_1.revise_trust(revision_fac_1);
  auto expected_op_2 = trust_2.revise_trust(revision_fac_2);

  TrustedOpinion<TypeParam> expected_t_op_1{ expected_op_1, opinion_1 };
  TrustedOpinion<TypeParam> expected_t_op_2{ expected_op_2, opinion_2 };
  auto [result_1, result_2] = t_op_1.revise_trust(t_op_2);

  EXPECT_EQ(result_1, expected_t_op_1);
  EXPECT_EQ(result_2, expected_t_op_2);
}

}  // namespace subjective_logic