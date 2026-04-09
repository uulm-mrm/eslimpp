#include <chrono>
#include <cstdint>

#include "subjective_logic_lib/opinions/opinion.hpp"
namespace sl = subjective_logic;
using OpT = sl::OpinionNoBase<2, float>;

int main(int argc, char** argv)
{
  constexpr long long n_runs{ 1LL << 20 };

  OpT op_a{ 0.2, 0.4 };
  OpT op_b{ 0.5, 0.1 };
  OpT op_c;
  asm volatile("" : : "r,m"(op_c) : "memory");
  float proj{};
  asm volatile("" : : "r,m"(proj) : "memory");

  std::cout << "running an experiment with " << n_runs << " runs" << std::endl;

  auto start = std::chrono::system_clock::now();

  for (std::size_t idx{ 0 }; idx < n_runs; ++idx)
  {
    op_c = op_a.cum_fuse(op_b);
    // proj = op_a.getBinomialProjection();
  }

  auto end = std::chrono::system_clock::now();

  std::cout << "last op: " << op_c << std::endl;
  // std::cout << "last proj: " << proj << std::endl;
  std::chrono::duration<double, std::nano> diff_time = end - start;
  std::cout << "time float: " << diff_time << std::endl;
  std::cout << "avg time float: " << (diff_time / n_runs) << std::endl;

  auto op_a_quant = op_a.get_quantized();
  auto op_b_quant = op_b.get_quantized();
  decltype(op_a_quant) op_c_quant;
  asm volatile("" : : "r,m"(op_c_quant) : "memory");
  float proj2{};
  asm volatile("" : : "r,m"(proj2) : "memory");

  constexpr bool test_type =
      std::is_same_v<decltype(op_a_quant),
                     subjective_logic::OpinionNoBase<2, subjective_logic::LimitedFloat<float, 0.0f, 1.0f>>>;
  static_assert(test_type);

  start = std::chrono::system_clock::now();

  for (std::size_t idx{ 0 }; idx < n_runs; ++idx)
  {
    op_c_quant = op_a_quant.cum_fuse(op_b_quant);
    // proj2 = op_a_quant.getBinomialProjection().as_limit_type();
  }

  end = std::chrono::system_clock::now();

  // std::cout << "last proj: " << proj2 << std::endl;
  std::cout << "last op: " << op_c_quant << std::endl;
  diff_time = end - start;
  std::cout << "time quant: " << diff_time << std::endl;
  std::cout << "avg time quant: " << (diff_time / n_runs) << std::endl;

  start = std::chrono::system_clock::now();

  for (std::size_t idx{ 0 }; idx < n_runs; ++idx)
  {
  }

  end = std::chrono::system_clock::now();
  diff_time = end - start;
  std::cout << "time ref: " << diff_time << std::endl;
  std::cout << "avg time ref: " << (diff_time / n_runs) << std::endl;
}

//
// namespace sl = subjective_logic;
// using Variable = sl::OpinionNoBase<2, float>;
//
// void printVariable(Variable a, std::string pretext = {})
// {
//   std::cout << pretext << a << std::endl;
//   //  std::cout << a.belief_mass_ << " | " << a.uncertainty_mass_ << std::endl;
// }
//
// int main(int argc, char* argv[])
// {
//   Variable A{ 0.7, 0.30 };
//   Variable B{ 0.1, 0.90 };
//
//   if (argc >= 5)
//   {
//     A = Variable{ std::stof(argv[1]), std::stof(argv[2]) };
//     B = Variable{ std::stof(argv[3]), std::stof(argv[4]) };
//   }
//
//   printVariable(A, "A: ");
//   printVariable(B, "B: ");
//
//   auto test = A.cc_fuse(B);
//   printVariable(test, "cc:");
//   //  std::cout << "test: " << test.belief_mass_ << " | " << test.uncertainty_mass_ << " | p: " <<
//   test.getProbability()
//   //            << std::endl;
//   //  Opinion M_l_a{ 0.8, 0.1 };
//   //  Opinion M_c_a{ 0.45, 0.1 };
//   //
//   //  Opinion M_l_b{ 0.8, 0.1 };
//   //  Opinion M_c_b{ 0.1, 0.1 };
//   //
//   //  std::cout << "harmony: " << M_l_b.harmony(M_c_b) << std::endl;
//   //  //  std::cout << "harmony dis: " << M_l_b.harmony_dis(M_c_b) << std::endl;
//   //  std::cout << "conflict: " << M_l_b.conflict(M_c_b) << std::endl;
//   //  Opinion bel_fuse{ M_l_b.belief_const_fuse(M_c_b) };
//   //  std::cout << "bel fusion: " << bel_fuse.belief_mass_ << " | " << bel_fuse.uncertainty_mass_ << std::endl;
//   //  std::cout << "bel projected: " << bel_fuse.getProjection(0.5) << std::endl;
//   //
//   //  Opinion cum_l = M_l_a.cum_fuse(M_l_b);
//   //  Opinion cum_c = M_c_a.cum_fuse(M_c_b);
//   //
//   //  Opinion bel_a = M_l_a.belief_const_fuse(M_c_a);
//   //  Opinion bel_b = M_l_b.belief_const_fuse(M_c_b);
//   //
//   //  printVariable(bel_a);
//   //  printVariable(bel_b);
//   //  printVariable(cum_l);
//   //  printVariable(cum_c);
//   //
//   //  std::cout << "\n\n";
//   //
//   //  Opinion cum_bel{ bel_a.cum_fuse(bel_b) };
//   //  printVariable(cum_bel);
//   //  std::cout << "cum_bel projected: " << cum_bel.getProjection(0.5) << std::endl;
//   //
//   //  Opinion bel_cum{ cum_l.belief_const_fuse(cum_c) };
//   //  printVariable(bel_cum);
//   //  std::cout << "bel_cum projected: " << bel_cum.getProjection(0.5) << std::endl;
//
//   //  Opinion a{.5, .5};
//   //  Opinion b{.45, .5};
//   //
//   //  Opinion cum_a{a};
//   //  Opinion cum_b{b};
//   //
//   //  Opinion bel_fuse = a.belief_const_fuse(b);
//   //  printVariable(bel_fuse);
//   //
//   //  Opinion bel_cum_fused;
//   //  bel_cum_fused.cum_fuse_(a.belief_const_fuse(b));
//   //  printVariable(bel_cum_fused);
//   //
//   //  std::cout << "loop\n";
//   //  constexpr std::size_t N_RUNS{10};
//   //  for (std::size_t i{0}; i<N_RUNS; ++i)
//   //  {
//   //    std::cout << "\n";
//   //    cum_a.cum_fuse_(a);
//   //    printVariable(cum_a);
//   //    cum_b.cum_fuse_(b);
//   //    printVariable(cum_b);
//   //
//   //    std::cout << "fusion\n";
//   //    Opinion bel_fuse = cum_a.belief_const_fuse(cum_b);
//   //    printVariable(bel_fuse);
//   //
//   //    bel_cum_fused.cum_fuse_(a.belief_const_fuse(b));
//   //    printVariable(bel_cum_fused);
//   //  }
//
//   return 0;
// }
