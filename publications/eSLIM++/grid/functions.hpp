#include <chrono>
#include <vector>

#include "subjective_logic_lib/opinions/opinion_no_base.hpp"

// ok for the sake of simplicity in examples
using namespace std::chrono;
using TimeDiff = duration<std::uint64_t, std::nano>;
using TimeDiffs = std::vector<duration<std::uint64_t, std::nano>>;

using Opinion = subjective_logic::OpinionNoBase<2,float>;
TimeDiffs run_gpu_assessment(std::size_t n_ops, std::size_t n_runs, const std::vector<Opinion>& sensor_a, const std::vector<Opinion>& sensor_b);
TimeDiffs run_cpu_assessment(std::size_t n_ops, std::size_t n_rund, std::vector<Opinion> sensor_a, std::vector<Opinion> sensor_b);

TimeDiffs run_cpu_assessment_fzi(std::size_t n_ops, std::size_t n_rund, std::vector<Opinion> sensor_a, std::vector<Opinion> sensor_b);
TimeDiffs run_dst_assessment_heudiasyc(std::size_t n_ops, std::size_t n_rund, std::vector<Opinion> sensor_a, std::vector<Opinion> sensor_b);
