#include "gtest/gtest.h"

#include "subjective_logic_lib/types/cuda_compatible_array.hpp"

namespace subjective_logic
{

TEST(ArrayTest, ValuesCtor)
{
  Array<2, float>{ 1, 2 };
  Array<2, float>(1.F, 2.F);
}

using TestTypes = ::testing::Types<Array<3, float>, Array<6, float>, Array<3, double>, Array<6, double> >;

template <typename ArrayT>
class ArrayTest : public ::testing::Test
{
protected:
  static constexpr std::size_t N = ArrayT::size();
  using T = ArrayT::value_type;

  void SetUp() override
  {
    for (std::size_t idx{ 0 }; idx < this->test.size(); ++idx)
    {
      this->test[idx] = static_cast<T>(idx);
    }
  }

  ArrayT test;
};

TYPED_TEST_SUITE(ArrayTest, TestTypes);

TYPED_TEST(ArrayTest, DefaultCtor)
{
  TypeParam test_array{};
  for (auto& entry : test_array)
  {
    EXPECT_FLOAT_EQ(entry, 0.F);
  }
}

TYPED_TEST(ArrayTest, DefaultValueCtor)
{
  typename TypeParam::value_type default_value{ 0.4 };
  TypeParam test_array{ default_value };
  for (auto& entry : test_array)
  {
    EXPECT_FLOAT_EQ(entry, default_value);
  }
}

TYPED_TEST(ArrayTest, CopyCtor)
{
  typename TypeParam::value_type default_value = 0.3;
  TypeParam expected_values;
  expected_values.fill(default_value);

  TypeParam test_array{ expected_values };
  for (auto& entry : test_array)
  {
    EXPECT_FLOAT_EQ(entry, default_value);
  }
}

TYPED_TEST(ArrayTest, StdArrayCtor)
{
  typename TypeParam::value_type default_entry = 0.25;
  std::array<typename TypeParam::value_type, TypeParam::size()> test_array;
  test_array.fill(default_entry);

  TypeParam test_copy{ test_array };
  for (auto& entry : test_copy)
  {
    EXPECT_FLOAT_EQ(entry, default_entry);
  }
}

TYPED_TEST(ArrayTest, IteratorContainerSize)
{
  std::size_t count{ 0 };
  for (auto& entry : this->test)
  {
    // by this the iterator is actually tested, in order to reduce warnings, the variable entry is used
    count += (entry >= 0) or (entry < 0);
  }
  EXPECT_EQ(count, this->N);
}

template <typename ArrayT>
void iterator_test_function(ArrayT& test_array)
{
  // the value at a certain index is always the index itself for testing reasons.
  auto test_begin = test_array.begin();
  EXPECT_FLOAT_EQ(*test_begin, 0);

  auto test_end = test_array.end();
  auto dist = distance(test_begin, test_end);
  EXPECT_EQ(dist, test_array.size());

  dist = distance(test_array.rbegin(), test_array.rend());
  EXPECT_EQ(dist, test_array.size());

  auto test_p1 = ++test_array.begin();
  EXPECT_EQ(*test_p1, 1);
  EXPECT_EQ(test_p1, ++test_array.begin());

  auto test_p2 = test_array.begin()++;
  EXPECT_EQ(*test_p2, 0);
  EXPECT_EQ(test_p2, test_array.begin());

  auto test_m1 = --test_array.end();
  EXPECT_EQ(*test_m1, test_array.size() - 1);
  EXPECT_EQ(test_m1, --test_array.end());

  auto test_m2 = test_array.end()--;
  EXPECT_EQ(test_m2, test_array.end());

  auto test_it = test_array.begin();
  for (std::size_t idx{ 0 }; idx < test_array.size(); ++idx)
  {
    EXPECT_FLOAT_EQ(*test_it, idx);
    ++test_it;
  }

  test_it = test_array.end();
  test_it--;
  for (int idx{ static_cast<int>(test_array.size()) - 1 }; idx >= 0; --idx)
  {
    EXPECT_FLOAT_EQ(*test_it, idx);
    --test_it;
  }

  auto test_rit = test_array.rbegin();
  for (int idx{ static_cast<int>(test_array.size()) - 1 }; idx >= 0; --idx)
  {
    EXPECT_FLOAT_EQ(*test_rit, idx);
    ++test_rit;
  }

  test_rit = test_array.rend();
  test_rit--;
  for (std::size_t idx{ 0 }; idx < test_array.size(); ++idx)
  {
    EXPECT_FLOAT_EQ(*test_rit, idx);
    --test_rit;
  }
}

TYPED_TEST(ArrayTest, Iterators)
{
  iterator_test_function(this->test);
  iterator_test_function(reinterpret_cast<const TypeParam&>(this->test));
}

TYPED_TEST(ArrayTest, Accessors)
{
  EXPECT_FLOAT_EQ(this->test.front(), static_cast<typename TypeParam::value_type>(0.));
  EXPECT_FLOAT_EQ(this->test.back(), static_cast<typename TypeParam::value_type>(this->test.size() - 1));

  EXPECT_FLOAT_EQ(reinterpret_cast<const TypeParam&>(this->test).front(),
                  static_cast<typename TypeParam::value_type>(0.));
  EXPECT_FLOAT_EQ(reinterpret_cast<const TypeParam&>(this->test).back(),
                  static_cast<typename TypeParam::value_type>(this->test.size() - 1));
}

TYPED_TEST(ArrayTest, OperatorPlus)
{
  typename TypeParam::value_type offset{ 2 };
  TypeParam offset_array{ 2 };

  auto copy1 = this->test;
  auto copy2 = this->test;

  auto test1 = this->test + offset;
  auto test2 = this->test + offset_array;
  auto test3 = offset + this->test;
  auto test4 = offset_array + this->test;
  copy1 += offset;
  copy2 += offset_array;

  for (std::size_t idx{ 0 }; idx < TypeParam::size(); ++idx)
  {
    EXPECT_FLOAT_EQ(this->test[idx] + offset, test1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] + offset, test2[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] + offset, test3[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] + offset, test4[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] + offset, copy1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] + offset, copy2[idx]);
  }
}

TYPED_TEST(ArrayTest, OperatorMinus)
{
  typename TypeParam::value_type offset{ 2 };
  TypeParam offset_array{ 2 };

  auto copy1 = this->test;
  auto copy2 = this->test;

  auto test1 = this->test - offset;
  auto test2 = this->test - offset_array;
  auto test3 = offset - this->test;
  auto test4 = offset_array - this->test;
  copy1 -= offset;
  copy2 -= offset_array;

  for (std::size_t idx{ 0 }; idx < TypeParam::size(); ++idx)
  {
    EXPECT_FLOAT_EQ(this->test[idx] - offset, test1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] - offset, test2[idx]);
    EXPECT_FLOAT_EQ(offset - this->test[idx], test3[idx]);
    EXPECT_FLOAT_EQ(offset - this->test[idx], test4[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] - offset, copy1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] - offset, copy2[idx]);
  }
}

TYPED_TEST(ArrayTest, OperatorMultiply)
{
  typename TypeParam::value_type offset{ 2 };
  TypeParam offset_array{ 2 };

  auto copy1 = this->test;
  auto copy2 = this->test;

  auto test1 = this->test * offset;
  auto test2 = this->test * offset_array;
  auto test3 = offset * this->test;
  auto test4 = offset_array * this->test;
  copy1 *= offset;
  copy2 *= offset_array;

  for (std::size_t idx{ 0 }; idx < TypeParam::size(); ++idx)
  {
    EXPECT_FLOAT_EQ(this->test[idx] * offset, test1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] * offset, test2[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] * offset, test3[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] * offset, test4[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] * offset, copy1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] * offset, copy2[idx]);
  }
}

TYPED_TEST(ArrayTest, OperatorDivide)
{
  typename TypeParam::value_type offset{ 2 };
  TypeParam offset_array{ 2 };

  auto copy1 = this->test;
  auto copy2 = this->test;

  auto test1 = this->test / offset;
  auto test2 = this->test / offset_array;
  auto test3 = offset / this->test;
  auto test4 = offset_array / this->test;
  copy1 /= offset;
  copy2 /= offset_array;

  for (std::size_t idx{ 0 }; idx < TypeParam::size(); ++idx)
  {
    EXPECT_FLOAT_EQ(this->test[idx] / offset, test1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] / offset, test2[idx]);
    if (idx != 0)
    {
      EXPECT_FLOAT_EQ(offset / this->test[idx], test3[idx]);
      EXPECT_FLOAT_EQ(offset / this->test[idx], test4[idx]);
    }
    EXPECT_FLOAT_EQ(this->test[idx] / offset, copy1[idx]);
    EXPECT_FLOAT_EQ(this->test[idx] / offset, copy2[idx]);
  }
}
TYPED_TEST(ArrayTest, Sum)
{
  auto sum = this->test.sum();
  auto expected_result = this->test.size() * (this->test.size() - 1) * 0.5;
  EXPECT_FLOAT_EQ(sum, expected_result);
}

}  // namespace subjective_logic
