#include "functions.hpp"
#include <iostream>

#include "mass_function.hpp"
#include "mobius_inversion.hpp"

using namespace efficient_DST;

TimeDiffs run_dst_assessment_heudiasyc(const std::size_t n_ops, const std::size_t n_runs, const std::vector<Opinion> sensor_a, const std::vector<Opinion> sensor_b) {
  constexpr std::size_t N=2;
  using MassFunction = mass_function<N,float>;
  using ConjComposition = conjunctive_decomposition_vector<N>;

  TimeDiffs runtimes(n_runs);
  std::size_t map_size_byte = n_ops * sizeof(MassFunction);

  std::string labels[] = {"occupied", "free"};
  sample_space<N> outcomes(labels);
  auto defaultMass = MassFunction{outcomes};

  std::vector<MassFunction> sensor_a_converted;
  sensor_a_converted.reserve(n_ops);
  std::vector<MassFunction> sensor_b_converted;
  sensor_b_converted.reserve(n_ops);


  for (std::size_t i = 0; i < n_ops; i++) {
    ConjComposition w_a(outcomes);
    ConjComposition w_b(outcomes);
    for (std::size_t idx{0}; idx < N; ++idx) {
      if (std::abs(sensor_a[i].belief_mass(idx) - 1) < 1e-5) {
        w_a.assign({labels[idx]}, sensor_a[i].belief_mass(idx) * 0.99);
      }
      else {
        w_a.assign({labels[idx]}, sensor_a[i].belief_mass(idx));
      }
      if (std::abs(sensor_b[i].belief_mass(idx) - 1) < 1e-5) {
        w_b.assign({labels[idx]}, sensor_b[i].belief_mass(idx) * 0.99);
      }
      else {
        w_b.assign({labels[idx]}, sensor_b[i].belief_mass(idx));
      }
    }
    if (std::abs(sensor_a[i].uncertainty() - 1) < 1e-5) {
      w_a.assign_emptyset(sensor_a[i].uncertainty()* 0.99);
    }
    else {
      w_a.assign_emptyset(sensor_a[i].uncertainty());
    }
    if (std::abs(sensor_b[i].uncertainty() - 1) < 1e-5) {
      w_b.assign_emptyset(sensor_b[i].uncertainty()* 0.99);
    }
    else {
      w_b.assign_emptyset(sensor_b[i].uncertainty());
    }
    sensor_a_converted.push_back(MassFunction{w_a});
    sensor_b_converted.push_back(MassFunction{w_b});
  }



  std::vector<int> results(n_ops);
  for (std::size_t run{0}; run < n_runs; ++run) {
    std::vector<MassFunction> dest;
    dest.reserve(n_ops);
    auto start = system_clock::now();
    for (std::size_t i{0}; i< n_ops; ++i) {
      dest.push_back(sensor_a_converted[i].natural_fusion_with<up_inclusion<N>>(sensor_b_converted[i]));
      // To the time of writing, I (wodtko) do not understand which function allows set wise weight reading
      // respectively, the categorization is omitted, it takes some time anyway
    }
    auto end = system_clock::now();
    runtimes[run] = end - start;
  }


  std::vector<int> hist;
  hist.resize(4);
  for (auto const entry : results) {
    hist[entry] += 1;
  }
  double denom = hist[1] + hist[3];
  double score = hist[3] / denom;
  std::cout << "size of one map with " << n_ops << " elements is: " << map_size_byte / 1e6 << "MB + dynamically allocated memory with unknown size to the author" << std::endl;
  std::cout << "the self-assessment score is: " << score << std::endl;

  return runtimes;
}

