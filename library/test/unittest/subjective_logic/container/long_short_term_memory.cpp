#include "gtest/gtest.h"

#include "subjective_logic_lib/container/long_short_term_memory.hpp"
#include "subjective_logic_lib/opinions/opinion_no_base.hpp"
#include "subjective_logic_lib/opinions/opinion.hpp"

namespace subjective_logic::container
{
using TestTypes =
    ::testing::Types<OpinionNoBase<3, float>, OpinionNoBase<6, double>, Opinion<3, double>, Opinion<6, float> >;

template <typename OpinionT>
class LSTMemoryTest : public ::testing::Test
{
  using FloatT = typename OpinionT::FLOAT_t;
  static constexpr std::size_t N = OpinionT::SIZE;

protected:
  void SetUp() override
  {
  }
};

TYPED_TEST_SUITE(LSTMemoryTest, TestTypes);

TYPED_TEST(LSTMemoryTest, Ctor)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;

  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.5 };

  LSTMemory memory(5, threshold, discount, [](TypeParam a, TypeParam b) { return a.cum_fuse(b); });
  LSTMemory memory2(8, threshold, discount, [](TypeParam a, TypeParam b) { return a.bc_fuse(b); });
  LSTMemory memory3(8, threshold, discount, [](TypeParam a, TypeParam b) { return a.average_fuse(b); });
}

TYPED_TEST(LSTMemoryTest, FusionSelection)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;

  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.5 };

  LSTMemory memory(5, threshold, discount, [](TypeParam a, TypeParam b) { return a.cum_fuse(b); });
  LSTMemory memory2(8, threshold, discount, [](TypeParam a, TypeParam b) { return a.bc_fuse(b); });
  LSTMemory memory3(8, threshold, discount, [](TypeParam a, TypeParam b) { return a.average_fuse(b); });

  // test if execution works, not if the fusion result is
  TypeParam default_op{};
  default_op.belief_masses()[0] = 0.2;
  default_op.belief_masses()[1] = 0.6;
  memory.add(default_op);
  memory.add(default_op);
  EXPECT_EQ(memory.get_opinion(), default_op.cum_fuse(default_op));

  memory2.add(default_op);
  memory2.add(default_op);
  EXPECT_EQ(memory2.get_opinion(), default_op.bc_fuse(default_op));

  memory3.add(default_op);
  memory3.add(default_op);
  EXPECT_EQ(memory3.get_opinion(), default_op.average_fuse(default_op));
}

TYPED_TEST(LSTMemoryTest, Add)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };

  constexpr std::size_t n_test = 5;
  LSTMemory memory(n_test, threshold, discount, [](TypeParam a, TypeParam b) { return a.cum_fuse(b); });

  TypeParam a{};
  a.belief_masses()[0] = 0.5;
  memory.add(a);

  TypeParam expected = a;
  EXPECT_EQ(memory.get_opinion(), a);

  for (std::size_t idx{ 1 }; idx < n_test; ++idx)
  {
    memory.add(a);
    expected.cum_fuse_(a);
  }
  auto short_term_a = expected;
  EXPECT_EQ(memory.get_opinion(), expected);

  auto long_expected = a;
  memory.add(a);
  EXPECT_EQ(memory.get_opinion(), short_term_a.cum_fuse(long_expected));
  EXPECT_EQ(memory.get_long_opinion(), long_expected);

  memory.add(a);
  long_expected.trust_discount_(discount);
  long_expected.cum_fuse_(a);
  EXPECT_EQ(memory.get_opinion(), short_term_a.cum_fuse(long_expected));
  EXPECT_EQ(memory.get_long_opinion(), long_expected);
}

TYPED_TEST(LSTMemoryTest, AddWithReset)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };

  constexpr std::size_t n_test = 5;
  LSTMemory memory(n_test, threshold, discount, [](TypeParam a, TypeParam b) { return a.cum_fuse(b); });

  TypeParam a{};
  a.belief_masses()[0] = 0.9;
  memory.add(a);

  TypeParam expected = a;

  for (std::size_t idx{ 1 }; idx < n_test; ++idx)
  {
    memory.add(a);
    expected.cum_fuse_(a);
  }
  auto short_term_a = expected;

  TypeParam b{};
  // all opinions have at least dim 2
  b.belief_masses()[1] = 0.9;

  // integer div indented
  for (std::size_t idx{ 0 }; idx < n_test / 2; ++idx)
  {
    memory.add(b);
  }
  EXPECT_EQ(memory.size(), n_test + n_test / 2);

  TypeParam last_output;
  last_output = memory.add(b);
  // conflict-based reset should be applied by now,
  // this depends on the threshold and opinions and was set such that it is triggered here
  // may change with different values in the future.
  EXPECT_EQ(memory.size(), memory.get_short_max_size());
  // conflict during last "add" leads to different output
  EXPECT_NE(memory.get_opinion(), last_output);

  // pushing another a from the short to the long buffer -> another reset expected
  last_output = memory.add(b);
  EXPECT_EQ(memory.size(), memory.get_short_max_size());
}

TYPED_TEST(LSTMemoryTest, SetShortMaxSize)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };

  constexpr std::size_t n_test = 5;
  LSTMemory memory(n_test, threshold, discount, [](TypeParam a, TypeParam b) { return a.cum_fuse(b); });

  TypeParam a{};
  a.belief_masses()[0] = 0.9;
  memory.add(a);

  TypeParam expected = a;

  for (std::size_t idx{ 1 }; idx < n_test; ++idx)
  {
    memory.add(a);
    expected.cum_fuse_(a);
  }
  auto short_term_a = expected;

  memory.add(a);
  EXPECT_EQ(memory.size(), n_test + 1);

  // reset long-term mem during short term resize
  // size after reset corresponds to the number of short-term opinions prior to change
  memory.set_short_max_size(n_test + 1);
  EXPECT_EQ(memory.size(), n_test);

  memory.add(a);
  // no reset long-term mem during short term resizes as no actual change
  memory.set_short_max_size(n_test + 1);
  EXPECT_EQ(memory.size(), n_test + 1);

  memory.set_short_max_size(n_test - 1);
  EXPECT_EQ(memory.size(), n_test - 1);
}

TYPED_TEST(LSTMemoryTest, Reset)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };

  constexpr std::size_t n_test = 5;
  LSTMemory memory(n_test, threshold, discount, [](TypeParam a, TypeParam b) { return a.cum_fuse(b); });

  TypeParam a{};
  a.belief_masses()[0] = 0.5;
  memory.add(a);

  TypeParam expected = a;
  for (std::size_t idx{ 1 }; idx < n_test; ++idx)
  {
    memory.add(a);
    expected.cum_fuse_(a);
  }
  memory.reset_long_memory();
  EXPECT_EQ(memory.get_opinion(), expected);

  auto long_expected = a;
  auto short_term_a = expected;
  memory.add(a);
  EXPECT_EQ(memory.get_opinion(), short_term_a.cum_fuse(long_expected));

  memory.reset_long_memory();
  EXPECT_EQ(memory.get_opinion(), expected);

  memory.reset();
  EXPECT_EQ(memory.get_opinion(), TypeParam::VacuousBeliefOpinion());
  EXPECT_EQ(memory.get_long_opinion(), TypeParam::VacuousBeliefOpinion());
}

}  // namespace subjective_logic::container