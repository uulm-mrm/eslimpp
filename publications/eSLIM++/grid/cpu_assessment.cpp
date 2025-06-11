#include "functions.hpp"
#include <iostream>


TimeDiffs run_cpu_assessment(const std::size_t n_ops, const std::size_t n_runs, const std::vector<Opinion> sensor_a, const std::vector<Opinion> sensor_b) {
  TimeDiffs runtimes(n_runs);
  std::size_t map_size_byte = n_ops * sizeof(Opinion);
  std::cout << "size of single subjective_logic_lib::OpinionNoBase<2,float>: " << sizeof(Opinion) << std::endl;
  std::vector<Opinion> dest(n_ops);
  std::vector<int> results(n_ops);


  for (std::size_t run{0}; run < n_runs; ++run) {
    auto start = system_clock::now();
    for (std::size_t i{0}; i< n_ops; ++i) {
      dest[i] = sensor_a[i].cum_fuse(sensor_b[i]);
      if (dest[i].uncertainty() > 0.5) {
        results[i] = 0;
      }
      else {
        Opinion::FLOAT_t prob = dest[i].getBinomialProjection();
        if (prob > 0.7) {
          results[i] = 1;
        }
        else if (prob < 0.3) {
          results[i] = 2;
        }
        else {
          results[i] = 3;
        }
      }

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
  std::cout << "size of one map with " << n_ops << " elements is: " << map_size_byte / 1e6 << "MB" << std::endl;
  std::cout << "the self-assessment score is: " << score << std::endl;

  return runtimes;
}

