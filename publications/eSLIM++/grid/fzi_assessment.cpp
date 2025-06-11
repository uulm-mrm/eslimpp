#include "functions.hpp"
#include <iostream>

#include "subj/MultinomialOpinion.h"
#include "subj/Operators.h"

TimeDiffs run_cpu_assessment_fzi(const std::size_t n_ops, const std::size_t n_runs, const std::vector<Opinion> sensor_a, const std::vector<Opinion> sensor_b) {
  TimeDiffs runtimes(n_runs);
  std::vector sensor_a_converted(n_ops, subj::MultinomialOpinion(Opinion::SIZE));
  std::vector sensor_b_converted(n_ops, subj::MultinomialOpinion(Opinion::SIZE));


  for (std::size_t i = 0; i < n_ops; i++) {
    std::vector<double> converted_belief_a;
    converted_belief_a.reserve(Opinion::SIZE);
    std::vector<double> converted_belief_b;
    converted_belief_b.reserve(Opinion::SIZE);
    for (std::size_t j = 0; j < Opinion::SIZE; j++) {
      converted_belief_a.push_back(sensor_a[i].belief_masses()[j]);
      converted_belief_b.push_back(sensor_b[i].belief_masses()[j]);
    }
    sensor_a_converted[i].updateBelief(converted_belief_a);
    sensor_a_converted[i].updateUncertainty(1.0 - sensor_a[i].belief_masses().sum());
    sensor_b_converted[i].updateBelief(converted_belief_b);
    sensor_b_converted[i].updateUncertainty(1.0 - sensor_b[i].belief_masses().sum());
  }

  std::size_t map_size_byte = sizeof(subj::MultinomialOpinion);
  std::cout << "size of single subj::MultinomialOpinion: " << sizeof(subj::MultinomialOpinion)  << " + dynamically allocated mem" << std::endl;
  // add entries of belief vector which are not part of the sizeof(...) before
  map_size_byte += sizeof(double) * sensor_a_converted[0].belief().size();
  // add entries of baseRate vector which are not part of the sizeof(...) before
  map_size_byte += sizeof(double) * sensor_a_converted[0].baseRate().size();
  std::cout << "total size of single subj::MultinomialOpinion: " << map_size_byte << std::endl;
  map_size_byte *= n_ops;


  std::vector dest(n_ops, subj::MultinomialOpinion(Opinion::SIZE));
  std::vector<int> results(n_ops);


  for (std::size_t run{0}; run < n_runs; ++run) {
    auto start = system_clock::now();
    for (std::size_t i{0}; i< n_ops; ++i) {
      dest[i] = subj::cbf(sensor_a_converted[i], sensor_b_converted[i]);
      dest[i].updateUncertainty(1.0 - dest[i].belief()[0] - dest[i].belief()[1]);
      if (dest[i].uncertainty() > 0.5) {
        results[i] = 0;
      }
      else {
        typename Opinion::FLOAT_t prob = dest[i].projection()[0];
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
  std::cout << "size of one map with " << n_ops << " elements is: " << map_size_byte / 1e6 << "MB and is scattered (multiple Heap allocations)" << std::endl;
  std::cout << "the self-assessment score is: " << score << std::endl;

  return runtimes;
}

