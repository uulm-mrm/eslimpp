#include <iostream>
#include <algorithm>

#include "gtest/gtest.h"

#include "subjective_logic_lib/multi_source/trust_revision_operators.hpp"

namespace subjective_logic::multisource
{

using TestTypes = ::testing::Types<OpinionNoBase<2, float>,
                                   OpinionNoBase<3, float>,
                                   OpinionNoBase<6, float>,
                                   OpinionNoBase<2, double>,
                                   OpinionNoBase<3, double>,
                                   OpinionNoBase<6, double>>;

template <typename OpinionT>
class MultiSourceNoBaseTrustRevisionTest : public ::testing::Test
{
  //    using FloatT = typename OpinionT::FLOAT_t;
  //    static constexpr std::size_t N = OpinionT::SIZE;
};

TYPED_TEST_SUITE(MultiSourceNoBaseTrustRevisionTest, TestTypes);

TYPED_TEST(MultiSourceNoBaseTrustRevisionTest, MaxConflictMaxDim)
{
}
}  // namespace subjective_logic::multisource