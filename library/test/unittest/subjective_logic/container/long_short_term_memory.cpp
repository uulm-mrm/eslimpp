#include "gtest/gtest.h"

#include "subjective_logic_lib/container/long_short_term_memory.hpp"
#include "subjective_logic_lib/multi_source/fusion_operators.hpp"
#include "subjective_logic_lib/opinions/opinion_no_base.hpp"
#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/types/fusion_types.hpp"

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

  LSTMemory memory(5, threshold, discount, subjective_logic::FusionType::CUMULATIVE);
  LSTMemory memory2(8, threshold, discount, subjective_logic::FusionType::BELIEF_CONSTRAINT);
  LSTMemory memory3(8, threshold, discount, subjective_logic::FusionType::AVERAGE);
}

TYPED_TEST(LSTMemoryTest, FusionSelection)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;

  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.5 };

  LSTMemory memory(5, threshold, discount, subjective_logic::FusionType::CUMULATIVE);
  LSTMemory memory2(8, threshold, discount, subjective_logic::FusionType::BELIEF_CONSTRAINT);
  LSTMemory memory3(8, threshold, discount, subjective_logic::FusionType::AVERAGE);

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
  LSTMemory memory(n_test, threshold, discount, subjective_logic::FusionType::CUMULATIVE);

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
  LSTMemory memory(n_test, threshold, discount, subjective_logic::FusionType::CUMULATIVE);

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
  EXPECT_TRUE(memory.is_last_conflicted());
  // conflict during last "add" leads to different output
  EXPECT_NE(memory.get_opinion(), last_output);

  // due to internal short-term memory conflict handling, no further resets are expected
  last_output = memory.add(b);
  EXPECT_FALSE(memory.is_last_conflicted());
}

TYPED_TEST(LSTMemoryTest, SetShortMaxSize)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };

  constexpr std::size_t n_test = 5;
  LSTMemory memory(n_test, threshold, discount, subjective_logic::FusionType::CUMULATIVE);

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
  LSTMemory memory(n_test, threshold, discount, subjective_logic::FusionType::CUMULATIVE);

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

TYPED_TEST(LSTMemoryTest, IdentityAverageFusion)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  LSTMemory mem(n_st, threshold, discount, FusionType::AVERAGE);

  TypeParam a{};
  a.belief_masses()[0] = 0.9;

  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(a);
  }

  EXPECT_EQ(mem.get_long_opinion(), a);
}

TYPED_TEST(LSTMemoryTest, IdentityWeightedFusion)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  LSTMemory mem(n_st, threshold, discount, FusionType::WEIGHTED);

  TypeParam a{};
  a.belief_masses()[0] = 0.9;

  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(a);
  }

  EXPECT_EQ(mem.get_long_opinion(), a);
}

TYPED_TEST(LSTMemoryTest, VacuousABF)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  LSTMemory mem(n_st, threshold, discount, FusionType::AVERAGE);
  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(TypeParam{});
  }
  EXPECT_EQ(mem.get_opinion(), TypeParam{});
}

TYPED_TEST(LSTMemoryTest, VacuousCBF)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  LSTMemory mem(n_st, threshold, discount, FusionType::CUMULATIVE);
  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(TypeParam{});
  }
  EXPECT_EQ(mem.get_opinion(), TypeParam{});
}

TYPED_TEST(LSTMemoryTest, VacuousWBF)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  LSTMemory mem(n_st, threshold, discount, FusionType::WEIGHTED);
  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(TypeParam{});
  }
  EXPECT_EQ(mem.get_opinion(), TypeParam{});
}

TYPED_TEST(LSTMemoryTest, DogmaticABF)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  TypeParam a{};
  a.belief_masses()[0] = 1.0;

  LSTMemory mem(n_st, threshold, discount, FusionType::AVERAGE);
  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(a);
  }
  EXPECT_EQ(mem.get_opinion(), a);
}

TYPED_TEST(LSTMemoryTest, DogmaticCBF)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  TypeParam a{};
  a.belief_masses()[0] = 1.0;

  LSTMemory mem(n_st, threshold, discount, FusionType::CUMULATIVE);
  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(a);
  }
  EXPECT_EQ(mem.get_opinion(), a);
}

TYPED_TEST(LSTMemoryTest, DogmaticWBF)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  TypeParam a{};
  a.belief_masses()[0] = 1.0;

  LSTMemory mem(n_st, threshold, discount, FusionType::WEIGHTED);
  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(a);
  }
  EXPECT_EQ(mem.get_opinion(), a);
}

TYPED_TEST(LSTMemoryTest, OneDogmaticABF)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.5 };
  constexpr double discount{ 0.8 };
  constexpr std::size_t n_st = 1;
  constexpr std::size_t n_ops = 20;

  TypeParam a{};
  a.belief_masses()[0] = 1.0;

  TypeParam b{};
  b.belief_masses()[1] = 0.5;

  LSTMemory mem(n_st, threshold, discount, FusionType::AVERAGE);
  for (std::size_t idx{ 0 }; idx < n_ops; ++idx)
  {
    mem.add(a);
    mem.add(b);
  }
  for (auto bm : mem.get_opinion().belief_masses())
  {
    EXPECT_FALSE(std::isnan(bm));
  }
}

TYPED_TEST(LSTMemoryTest, ShortTermReset)
{
  using LSTMemory = LongShortTermMemory<TypeParam>;
  constexpr double threshold{ 0.2 };
  constexpr double discount{ 0.9 };
  constexpr std::size_t n_st{ 10 };
  constexpr std::size_t num_ops{ 25 };

  TypeParam op_a{};
  op_a.belief_masses()[0] = 0.8;
  TypeParam op_b{};
  op_b.belief_masses()[1] = 0.8;

  // jump sequence to trigger reset
  std::vector<TypeParam> ops(2 * num_ops);
  for (std::size_t idx{ 0 }; idx < num_ops; ++idx)
  {
    ops[idx] = op_a;
    ops[idx + num_ops] = op_b;
  }

  LSTMemory avg_dc_mem(n_st, threshold, discount, FusionType::AVERAGE, true, true);
  LSTMemory fusion_mem(n_st, threshold, discount, FusionType::AVERAGE, true, false);

  bool reset_fusion{ false };
  bool reset_avg_dc{ false };

  auto validate_reset = [](LSTMemory mem, std::size_t idx, bool& reset) {
    if (!mem.is_last_conflicted())
    {
      return;
    }
    std::size_t expected_ST_size = idx - num_ops;

    EXPECT_EQ(expected_ST_size, mem.get_short_size());
    // sanity check reset location
    EXPECT_TRUE(idx >= num_ops and idx < num_ops + n_st);
    reset = true;
  };

  for (std::size_t idx{ 0 }; idx < 2 * num_ops; ++idx)
  {
    fusion_mem.add(ops[idx]);
    avg_dc_mem.add(ops[idx]);

    validate_reset(avg_dc_mem, idx, reset_avg_dc);
    validate_reset(fusion_mem, idx, reset_fusion);
  }

  // make sure reset occurred
  EXPECT_TRUE(reset_avg_dc);
  EXPECT_TRUE(reset_fusion);
}
}  // namespace subjective_logic::container