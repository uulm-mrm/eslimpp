#include "subjective_logic_lib/opinions/opinion.hpp"
#include <iostream>

namespace sl = subjective_logic;
using Variable = sl::OpinionNoBase<2, float>;

void printVariable(Variable a, std::string pretext = {})
{
  std::cout << pretext << a << std::endl;
  //  std::cout << a.belief_mass_ << " | " << a.uncertainty_mass_ << std::endl;
}

int main(int argc, char* argv[])
{
  Variable A{ 0.7, 0.30 };
  Variable B{ 0.1, 0.90 };

  if (argc >= 5)
  {
    A = Variable{ std::stof(argv[1]), std::stof(argv[2]) };
    B = Variable{ std::stof(argv[3]), std::stof(argv[4]) };
  }

  printVariable(A, "A: ");
  printVariable(B, "B: ");

  auto test = A.cc_fuse(B);
  printVariable(test, "cc:");
  //  std::cout << "test: " << test.belief_mass_ << " | " << test.uncertainty_mass_ << " | p: " << test.getProbability()
  //            << std::endl;
  //  Opinion M_l_a{ 0.8, 0.1 };
  //  Opinion M_c_a{ 0.45, 0.1 };
  //
  //  Opinion M_l_b{ 0.8, 0.1 };
  //  Opinion M_c_b{ 0.1, 0.1 };
  //
  //  std::cout << "harmony: " << M_l_b.harmony(M_c_b) << std::endl;
  //  //  std::cout << "harmony dis: " << M_l_b.harmony_dis(M_c_b) << std::endl;
  //  std::cout << "conflict: " << M_l_b.conflict(M_c_b) << std::endl;
  //  Opinion bel_fuse{ M_l_b.belief_const_fuse(M_c_b) };
  //  std::cout << "bel fusion: " << bel_fuse.belief_mass_ << " | " << bel_fuse.uncertainty_mass_ << std::endl;
  //  std::cout << "bel projected: " << bel_fuse.getProjection(0.5) << std::endl;
  //
  //  Opinion cum_l = M_l_a.cum_fuse(M_l_b);
  //  Opinion cum_c = M_c_a.cum_fuse(M_c_b);
  //
  //  Opinion bel_a = M_l_a.belief_const_fuse(M_c_a);
  //  Opinion bel_b = M_l_b.belief_const_fuse(M_c_b);
  //
  //  printVariable(bel_a);
  //  printVariable(bel_b);
  //  printVariable(cum_l);
  //  printVariable(cum_c);
  //
  //  std::cout << "\n\n";
  //
  //  Opinion cum_bel{ bel_a.cum_fuse(bel_b) };
  //  printVariable(cum_bel);
  //  std::cout << "cum_bel projected: " << cum_bel.getProjection(0.5) << std::endl;
  //
  //  Opinion bel_cum{ cum_l.belief_const_fuse(cum_c) };
  //  printVariable(bel_cum);
  //  std::cout << "bel_cum projected: " << bel_cum.getProjection(0.5) << std::endl;

  //  Opinion a{.5, .5};
  //  Opinion b{.45, .5};
  //
  //  Opinion cum_a{a};
  //  Opinion cum_b{b};
  //
  //  Opinion bel_fuse = a.belief_const_fuse(b);
  //  printVariable(bel_fuse);
  //
  //  Opinion bel_cum_fused;
  //  bel_cum_fused.cum_fuse_(a.belief_const_fuse(b));
  //  printVariable(bel_cum_fused);
  //
  //  std::cout << "loop\n";
  //  constexpr std::size_t N_RUNS{10};
  //  for (std::size_t i{0}; i<N_RUNS; ++i)
  //  {
  //    std::cout << "\n";
  //    cum_a.cum_fuse_(a);
  //    printVariable(cum_a);
  //    cum_b.cum_fuse_(b);
  //    printVariable(cum_b);
  //
  //    std::cout << "fusion\n";
  //    Opinion bel_fuse = cum_a.belief_const_fuse(cum_b);
  //    printVariable(bel_fuse);
  //
  //    bel_cum_fused.cum_fuse_(a.belief_const_fuse(b));
  //    printVariable(bel_cum_fused);
  //  }

  return 0;
}
