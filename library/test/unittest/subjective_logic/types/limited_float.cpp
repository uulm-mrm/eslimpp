#include "gtest/gtest.h"

#include "subjective_logic_lib/types/limited_float.hpp"

namespace subjective_logic
{

TEST(LimitedFloatTest, Traits)
{
  EXPECT_EQ(sizeof(LimitedFloat<double, 0.0, 1.0>), 1);
}

TEST(LimitedFloatTest, Ctor)
{
  LimitedFloat<double, 0.0, 1.0> lf{ 0.5 };
  EXPECT_EQ(lf.value, static_cast<std::uint8_t>(0.5 * std::numeric_limits<std::uint8_t>::max()));

  LimitedFloat<double, 2.0, 5.0> lf2{ 0.5 };
  EXPECT_EQ(lf2.value, 0);
}

TEST(LimitedFloatTest, Cast)
{
  double precision = 1.0 / 255;
  LimitedFloat<double, 0.0, 1.0> lf{ 0.5 };
  EXPECT_NEAR(static_cast<double>(lf), 0.5, precision);

  LimitedFloat<double, 0.0, 1.0> lf2{ 0.75 };
  EXPECT_NEAR(static_cast<double>(lf2), 0.75, precision);

  LimitedFloat<double, 1.0, 2.0> lf3{ 0.75 };
  EXPECT_NEAR(static_cast<double>(lf3), 1.0, precision);
}

TEST(LimitedFloatTest, Operators)
{
  // precision would actually state what the limited float can express.
  // however, due to different conversions, the actual error might be a bit higher
  // this is mostly due to putting the start value inside a limited float and
  // then calculate with the incorrect, approximated value afterward
  double precision = 1.2 / 255;

  constexpr double lower_bound = 0.0;
  constexpr double upper_bound = 1.0;

  constexpr double value = 0.5;
  constexpr double value2 = 0.2;
  constexpr double diff = 0.1;
  constexpr double diff2 = 2.1;

  using lFloat = LimitedFloat<double, lower_bound, upper_bound>;
  constexpr lFloat lf{ value };
  constexpr lFloat lf2{ value2 };

  auto test_plus = lf + diff;
  constexpr bool type_plus = std::is_same_v<decltype(test_plus), double>;
  static_assert(type_plus);
  EXPECT_NEAR(test_plus, value + diff, precision);
  test_plus = lf + diff2;
  EXPECT_NEAR(test_plus, value + diff2, precision);
  lFloat lf_plus{ lf + diff };
  EXPECT_NEAR(static_cast<double>(lf_plus), value + diff, precision);
  lf_plus = lf + diff2;
  EXPECT_NEAR(static_cast<double>(lf_plus), upper_bound, precision);

  lFloat lf_plus2{ lf + lf2 };
  EXPECT_NEAR(static_cast<double>(lf_plus2), value + value2, 2 * precision);
  double lf_plus2_float = lf + lf2;
  EXPECT_NEAR(lf_plus2_float, value + value2, 2 * precision);

  auto test_minus = lf - diff;
  constexpr bool type_minus = std::is_same_v<decltype(test_minus), double>;
  static_assert(type_minus);
  EXPECT_NEAR(test_minus, value - diff, precision);
  test_minus = lf - diff2;
  EXPECT_NEAR(test_minus, value - diff2, precision);
  lFloat lf_minus{ lf - diff };
  EXPECT_NEAR(static_cast<double>(lf_minus), value - diff, precision);
  lf_minus = lf - diff2;
  EXPECT_NEAR(static_cast<double>(lf_minus), lower_bound, precision);

  auto test_mult = lf * diff;
  constexpr bool type_mult = std::is_same_v<decltype(test_mult), double>;
  static_assert(type_mult);
  EXPECT_NEAR(test_mult, value * diff, precision);
  test_mult = lf * diff2;
  EXPECT_NEAR(test_mult, value * diff2, precision);
  lFloat lf_mult{ lf * diff };
  EXPECT_NEAR(static_cast<double>(lf_mult), value * diff, precision);
  lf_mult = lf * diff2;
  EXPECT_NEAR(static_cast<double>(lf_mult), upper_bound, precision);

  auto test_div = lf / diff;
  constexpr bool type_div = std::is_same_v<decltype(test_div), double>;
  static_assert(type_div);
  EXPECT_NEAR(test_div, value / diff, precision / diff);
  test_div = lf / diff2;
  EXPECT_NEAR(test_div, value / diff2, precision);
  lFloat lf_div{ lf / diff };
  EXPECT_NEAR(static_cast<double>(lf_div), upper_bound, precision);
  lf_div = lf / diff2;
  EXPECT_NEAR(static_cast<double>(lf_div), value / diff2, precision);
}

}  // namespace subjective_logic