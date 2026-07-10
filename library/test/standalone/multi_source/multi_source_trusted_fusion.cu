#include <iostream>
#include <chrono>
#include "subjective_logic_lib/opinions/opinion.hpp"
#include "subjective_logic_lib/opinions/trusted_opinion.hpp"
#include "subjective_logic_lib/multi_source/trusted_fusion_operators.hpp"

namespace sl = subjective_logic;
using Opinion = sl::Opinion<2,float>;
using TrustedOpinion = sl::TrustedOpinion<Opinion>;
constexpr Opinion trust{Opinion::BeliefType{0.,0.},Opinion::BeliefType{1.0,0.0}};
constexpr std::size_t n_opinions{1 << 20};
// constexpr std::size_t n_opinions{1 << 0};
constexpr std::size_t num_fuse_opinions{3};

constexpr sl::Array<1,sl::multisource::TrustedFusion::WeightedTypes> weights{sl::multisource::TrustedFusion::WeightedTypes{
  sl::RelationType::CONFLICT,
  sl::TrustRevisionType::SHARES,
  sl::ConflictType::AVERAGE,
  1.0
}};

#define CUDA_CHECK(call) do { cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error: %s (%d) ", cudaGetErrorString(err), err); exit(EXIT_FAILURE); \
}\
} while (0)


__global__ static void run_trusted_fusion(Opinion* a, Opinion* b, Opinion* c, Opinion* d, Opinion* dest) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_opinions) {return;}

  sl::Array<num_fuse_opinions,TrustedOpinion> topins{
    sl::TrustedOpinion{trust,a[idx]},
    sl::TrustedOpinion{trust,b[idx]},
    // sl::TrustedOpinion{trust,c[idx]},
    sl::TrustedOpinion{trust,d[idx]}
  };

  dest[idx] = sl::multisource::TrustedFusion::fuse_opinions(sl::FusionType::CUMULATIVE, weights, topins);
}

void check_run_trusted_fusion() {
  std::cout << "testing trusted fusion with " << num_fuse_opinions << " opinions " << std::to_string(n_opinions) << " times" << std::endl;
  std::vector<Opinion> input_a(n_opinions);
  std::vector<Opinion> input_b(n_opinions);
  std::vector<Opinion> input_c(n_opinions);
  std::vector<Opinion> input_d(n_opinions);
  for (std::size_t idx = 0; idx < n_opinions; ++idx) {
    // a uncert = 0.2
    input_a[idx].belief_masses()[0] = 0.8;
    // b uncert = 0.8
    input_b[idx].belief_masses()[1] = 0.2;
    // c uncert = 0.6
    input_c[idx].belief_masses()[0] = 0.2;
    input_c[idx].belief_masses()[1] = 0.2;
    // d uncert = 0.3
    input_c[idx].belief_masses()[0] = 0.5;
    input_c[idx].belief_masses()[1] = 0.2;

    // sum of uncert = 1.9
  }
  // average prior
  Opinion *op_a, *op_b, *op_c, *op_d;
  using DestType = Opinion;
  DestType *dest;
  cudaMalloc(&op_a, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_b, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_c, sizeof(Opinion)*n_opinions);
  cudaMalloc(&op_d, sizeof(Opinion)*n_opinions);
  cudaMalloc(&dest, sizeof(DestType)*n_opinions);

  cudaMemcpy(op_a, input_a.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_b, input_b.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_c, input_c.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);
  cudaMemcpy(op_d, input_d.data(), sizeof(Opinion)*n_opinions,cudaMemcpyHostToDevice);

  CUDA_CHECK(cudaDeviceSynchronize());
  auto start_time = std::chrono::high_resolution_clock::now();
  run_trusted_fusion<<<n_opinions,32,32>>>(op_a, op_b, op_c, op_d, dest);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();

  std::vector<DestType> dest_cpu;
  dest_cpu.resize(n_opinions);

  cudaMemcpy(dest_cpu.data(), dest, sizeof(DestType)*n_opinions,cudaMemcpyDeviceToHost);

  std::cout << dest_cpu[0] << std::endl;
  std::cout << "took " << std::chrono::duration<double, std::milli>(end_time-start_time) << std::endl;
  auto weight_vec = weights.as_vector();
  auto fused_opin = sl::multisource::TrustedFusion::fuse_opinions(
    sl::FusionType::CUMULATIVE,
    weight_vec,
    std::vector<TrustedOpinion>{
      {trust,input_a.front()},
      {trust,input_b.front()},
      // {trust,input_c.front()},
      {trust,input_d.front()},
    }) ;

  std::cout << "reference fused opinion: " << fused_opin << std::endl;

  cudaFree(&op_a);
  cudaFree(&op_b);
  cudaFree(&op_c);
  cudaFree(&op_d);
  cudaFree(&dest);
}
