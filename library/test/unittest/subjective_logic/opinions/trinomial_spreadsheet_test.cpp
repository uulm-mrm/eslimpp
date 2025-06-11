#include "gtest/gtest.h"

#include "subjective_logic_lib/opinions/opinion_no_base.hpp"

namespace subjective_logic
{

template <typename OpinionT>
OpinionT test_deduction(OpinionT x,
                        typename OpinionT::BeliefType base_x,
                        Array<OpinionT::SIZE, OpinionT> conditionals,
                        OpinionT expected_result)
{
  OpinionT result = x.deduction(base_x, conditionals);

  for (std::size_t idx{ 0 }; idx < OpinionT::SIZE; ++idx)
  {
    // given numbers in the example are rounded, respective deviations are expected
    EXPECT_NEAR(expected_result.belief_mass(idx), result.belief_mass(idx), 0.01);
  }
  EXPECT_NEAR(expected_result.uncertainty(), result.uncertainty(), 0.01);

  return result;
}

TEST(TrinomialSpreadsheetTest, JosangExampleTrinomialDeduction1)
{
  using Opinion = OpinionNoBase<3, double>;

  Opinion x{ 0.5, 0.1, 0.1 };
  Opinion::BeliefType base_x{ 0.1, 0.1, 0.8 };
  Array<3, Opinion> conditionals{ Opinion{ 0.0, 0.7, 0.1 }, Opinion{ 0.7, 0.0, 0.1 }, Opinion{ 0.1, 0.1, 0.2 } };

  Opinion expected_y{ 0.10171, 0.38171, 0.11 };

  test_deduction(x, base_x, conditionals, expected_y);
}

TEST(TrinomialSpreadsheetTest, JosangExampleTrinomialDeduction2)
{
  using Opinion = OpinionNoBase<3, double>;

  Opinion x{ 0., 0., 0. };
  Opinion::BeliefType base_x{ 0.2, 0.8, 0.0 };
  Array<3, Opinion> conditionals{ Opinion{ 0.4, 0.0, 0.0001 },
                                  Opinion{ 0.5, 0.4, 0.0001 },
                                  Opinion{ 0.0, 0.0, 0.0001 } };

  Opinion expected_y{ 0.48, 0.32, 0.0001 };

  test_deduction(x, base_x, conditionals, expected_y);
}

TEST(TrinomialSpreadsheetTest, JosangExampleTrinomialDeduction3)
{
  using Opinion = OpinionNoBase<3, double>;

  Opinion x{ 0., 0., 0. };
  Opinion::BeliefType base_x{ 0.2, 0.75, 0.05 };
  Array<3, Opinion> conditionals{ Opinion{ 0.4, 0.0, 0.0001 },
                                  Opinion{ 0.5, 0.4, 0.0001 },
                                  Opinion{ 0.0, 0.0, 0.0001 } };

  Opinion expected_y{ 0.455, 0.3, 0.0001 };

  test_deduction(x, base_x, conditionals, expected_y);
}

TEST(TrinomialSpreadsheetTest, JosangExampleTrinomialDeduction4)
{
  using Opinion = OpinionNoBase<3, double>;

  Opinion x{ 0., 0., 0. };
  Opinion::BeliefType base_x{ 0.2, 0.75, 0.05 };
  Array<3, Opinion> conditionals{ Opinion{ 0.2, 0.2, 0.2 }, Opinion{ 0.6, 0.1, 0.1 }, Opinion{ 0.1, 0.1, 0.6 } };

  Opinion expected_y{ 0.4125, 0.1, 0.12083 };

  test_deduction(x, base_x, conditionals, expected_y);
}

TEST(TrinomialSpreadsheetTest, JosangExampleTrinomialDeduction5)
{
  using Opinion = OpinionNoBase<3, double>;

  Opinion x{ 0.1, 0.2, 0.3 };
  Opinion::BeliefType base_x{ 0.2, 0.75, 0.05 };
  Array<3, Opinion> conditionals{ Opinion{ 0.2, 0.2, 0.2 }, Opinion{ 0.6, 0.1, 0.1 }, Opinion{ 0.1, 0.1, 0.6 } };

  Opinion expected_y{ 0.335, 0.11, 0.26833 };

  test_deduction(x, base_x, conditionals, expected_y);
}

TEST(TrinomialSpreadsheetTest, JosangExampleTrinomialDeduction6)
{
  using Opinion = OpinionNoBase<3, double>;

  Opinion x{ 0.1, 0.8, 0.1 };
  Opinion::BeliefType base_x{ 0.2, 0.75, 0.05 };
  Array<3, Opinion> conditionals{ Opinion{ 0.2, 0.2, 0.2 }, Opinion{ 0.6, 0.1, 0.1 }, Opinion{ 0.1, 0.1, 0.6 } };

  Opinion expected_y{ 0.51, 0.11, 0.16 };

  test_deduction(x, base_x, conditionals, expected_y);
}

TEST(TrinomialSpreadsheetTest, JosangExampleBinomialDeductionFig95)
{
  using Opinion = OpinionNoBase<2, double>;

  Opinion x{ 0., 0. };
  Opinion::BeliefType base_x{ 0.8, 0.2 };
  Array<2, Opinion> conditionals{
    Opinion{ 0.4, 0.5 },
    Opinion{ 0.0, 0.4 },
  };

  //  Opinion expected_book{0.32, 0.48};
  Opinion expected_y{ 0.266666666, 0.4 };

  Opinion y = test_deduction(x, base_x, conditionals, expected_y);

  Opinion binomial_test = x.deduction(base_x[0], conditionals[0], conditionals[1]);
  EXPECT_FLOAT_EQ(y.belief(), binomial_test.belief());
  EXPECT_FLOAT_EQ(y.disbelief(), binomial_test.disbelief());
  EXPECT_FLOAT_EQ(y.uncertainty(), binomial_test.uncertainty());
}

}  // namespace subjective_logic