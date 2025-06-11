#include <iostream>
#include "functions.hpp"

#include "subj/MultinomialOpinion.h"

std::size_t n_ops = 1000000;
std::size_t n_runs = 100;


std::tuple<TimeDiff, TimeDiff, TimeDiff, TimeDiff, TimeDiff> get_quantiles(TimeDiffs& diffs) {
  std::ranges::sort(diffs);
  const std::size_t lower_idx  = diffs.size() * 0.01;
  const std::size_t box_lower_idx  = diffs.size() * 0.25;
  const std::size_t median_idx = diffs.size() * 0.5;
  const std::size_t box_higher_idx = diffs.size() * 0.75;
  const std::size_t higher_idx = diffs.size() * 0.99;

  return {
    diffs[median_idx],
    diffs[box_lower_idx],
    diffs[box_higher_idx],
    diffs[higher_idx],
    diffs[lower_idx]
  };
}

int main( int argc, char ** argv )
{
  if (argc > 1) {
    n_ops = std::stoi(argv[1]);
  }
  if (argc > 2) {
    n_runs = std::stoi(argv[2]);
  }

  std::vector<Opinion> sensor_a(n_ops, Opinion(0,0));
  std::vector<Opinion> sensor_b(n_ops, Opinion(0,0));


  // both sensors the same
  for (std::size_t i{0}; i< 1000; ++i) {
    sensor_a[i] = Opinion(0.9, 0.0);
    sensor_b[i] = Opinion(0.9, 0.0);
  }

  // each sensor separately
  for (std::size_t i{2000}; i< 3000; ++i) {
    sensor_a[i] = Opinion(0.9, 0.0);
    sensor_b[i] = Opinion(0.0, 0.9);
  }

  std::cout << "conversion times are not accounted for in any approach, only the execution of fusion and categorization." << std::endl;
  std::cout << std::endl;

  TimeDiffs elapsed;
  std::tuple<TimeDiff, TimeDiff, TimeDiff, TimeDiff, TimeDiff> quantiles;

#ifndef NO_GPU
  std::cout << "running OursGPU:" << std::endl;
  elapsed = run_gpu_assessment(n_ops, n_runs, sensor_a, sensor_b);
  quantiles = get_quantiles(elapsed);
  std::cout << "gpu median runtime for " << n_ops << " calls: " << duration<double,std::milli>(std::get<0>(quantiles)) << "\n";
  std::cout << "output for tikz:\n";
  std::cout << "gpu: "
    << duration<double,std::milli>(std::get<0>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<1>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<2>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<3>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<4>(quantiles)).count() << " \n";
  std::cout << std::endl;
  std::cout << std::endl;
#else
  std::cout << "Cuda was deactivated or not found during compile time, thus, the respective evaluation is skipped.\n\n" << std::endl;
#endif


  std::cout << "running OursCPU:" << std::endl;
  elapsed = run_cpu_assessment(n_ops, n_runs, sensor_a, sensor_b);
  quantiles = get_quantiles(elapsed);
  std::cout << "cpu median runtime for " << n_ops << " calls: " << duration<double, std::milli>(std::get<0>(quantiles)) << "\n";
  std::cout << "output for tikz:\n";
  std::cout << "cpu: "
    << duration<double,std::milli>(std::get<0>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<1>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<2>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<3>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<4>(quantiles)).count() << " \n";
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "running FZI:" << std::endl;
  elapsed = run_cpu_assessment_fzi(n_ops, n_runs, sensor_a, sensor_b);
  quantiles = get_quantiles(elapsed);
  std::cout << "baseline impl" << std::endl;
  std::cout << "cpu median runtime for " << n_ops << " calls: " << duration<double, std::milli>(std::get<0>(quantiles)) << "\n";
  std::cout << "output for tikz:\n";
  std::cout << "fzi: "
    << duration<double,std::milli>(std::get<0>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<1>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<2>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<3>(quantiles)).count() << " "
    << duration<double,std::milli>(std::get<4>(quantiles)).count() << " \n";
  std::cout << std::endl;
  std::cout << std::endl;


#ifdef RUN_DST
  std::cout << "running DST (possible missuse of operators, not included in publication):" << std::endl;
  elapsed = run_dst_assessment_heudiasyc(n_ops, n_runs, sensor_a, sensor_b);
  quantiles = get_quantiles(elapsed);
  std::cout << "dst baseline impl" << std::endl;
  std::cout << "cpu median runtime for " << n_ops << " calls: " << duration<double, std::milli>(std::get<0>(quantiles)) << "\n";
  std::cout << std::endl;
  std::cout << std::endl;
#endif

  return 0;
}